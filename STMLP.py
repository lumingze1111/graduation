from logging import getLogger

import torch
import torch.nn as nn
import torch.nn.functional as F
from libcity.model import loss
from libcity.model.abstract_traffic_state_model import AbstractTrafficStateModel
import numpy as np
import scipy.sparse as sp
from scipy.sparse import linalg
class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """
    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x


class series_decomp(nn.Module):
    """
    Series decomposition block
    """
    def __init__(self, kernel_size=25):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean
def sym_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).astype(np.float32).todense()

def asym_adj(adj):
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1)).flatten()
    d_inv = np.power(rowsum, -1).flatten()
    d_inv[np.isinf(d_inv)] = 0.
    d_mat = sp.diags(d_inv)
    return d_mat.dot(adj).astype(np.float32).todense()

def calculate_normalized_laplacian(adj):
    """
    # L = D^-1/2 (D-A) D^-1/2 = I - D^-1/2 A D^-1/2
    # D = diag(A 1)
    :param adj:
    :return:
    """
    adj = sp.coo_matrix(adj)
    d = np.array(adj.sum(1))
    d_inv_sqrt = np.power(d, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    normalized_laplacian = sp.eye(adj.shape[0]) - adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()
    return normalized_laplacian

def calculate_scaled_laplacian(adj_mx, lambda_max=2, undirected=True):
    if undirected:
        adj_mx = np.maximum.reduce([adj_mx, adj_mx.T])
    lap = calculate_normalized_laplacian(adj_mx)
    if lambda_max is None:
        lambda_max, _ = linalg.eigsh(lap, 1, which='LM')
        lambda_max = lambda_max[0]
    lap = sp.csr_matrix(lap)
    m, _ = lap.shape
    identity = sp.identity(m, format='csr', dtype=lap.dtype)
    lap = (2 / lambda_max * lap) - identity
    return lap.astype(np.float32).todense()
class MlpBlock(nn.Module):
    def __init__(self, mlp_dim):
        super(MlpBlock, self).__init__()
        self.mlp_dim = mlp_dim
        self.fc1 = nn.Linear(self.mlp_dim, self.mlp_dim)
        self.fc2 = nn.Linear(self.mlp_dim, self.mlp_dim)

    def forward(self, x):
        y = self.fc1(x)
        y = F.gelu(y)
        return self.fc2(y)
    

class MixerBlock(nn.Module):
    def __init__(self, nodes_mlp_dim, length_mlp_dim):
        super(MixerBlock, self).__init__()
        self.nodes_mlp_dim = nodes_mlp_dim
        self.length_mlp_dim = length_mlp_dim
        self.layer_norm1 = nn.LayerNorm(normalized_shape=self.nodes_mlp_dim)  # You need to specify the shape for normalization
        self.layer_norm2 = nn.LayerNorm(normalized_shape=self.length_mlp_dim)  # You need to specify the shape for normalization
        self.token_mixing = MlpBlock(self.length_mlp_dim)  # MlpBlock needs to be implemented as described before
        self.channel_mixing = MlpBlock(self.nodes_mlp_dim)  # MlpBlock needs to be implemented as described before
        self.drop = nn.Dropout(p=0.3)
    def forward(self, x):
        y = self.layer_norm1(x)
        y = y.permute(0, 2, 1)  # Swaps axes in PyTorch
        # print("y.shape = ",y.shape)
        y = self.token_mixing(y)
        # y = self.drop(y)
        y = self.layer_norm2(y)
        y = y.permute(0, 2, 1)  # Swaps axes back to the original

        return x + self.channel_mixing(y)
import os
from datetime import datetime
class STMLP(AbstractTrafficStateModel):
    def __init__(self, config, data_feature):
        super().__init__(config, data_feature)
        self.num_nodes = data_feature.get('num_nodes')
        self.input_window = config.get('input_window')
        self.output_window = config.get('output_window')
        self.feature_dim = data_feature.get('feature_dim', 2)
        self.output_dim = self.data_feature.get('output_dim', 1)
        self.time_intervals = config.get('time_intervals')
        self._scaler = self.data_feature.get('scaler')
        self.adjtype = config.get('adjtype', "doubletransition")
        self.adj_mx = data_feature.get('adj_mx')
        self.cal_adj(self.adjtype)
        self.num_blocks = config.get('num_block')
        self.time_series_emb_dim = config.get('time_series_emb_dim')
        self.spatial_emb_dim = config.get('spatial_emb_dim')
        self.temp_dim_tid = config.get('temp_dim_tid')
        self.temp_dim_diw = config.get('temp_dim_diw')
        self.if_spatial = config.get('if_spatial')
        self.if_time_in_day = config.get('if_TiD')
        self.if_day_in_week = config.get('if_DiW')
        self.patch_len = config.get('patch_len')
        self.d_model = config.get('d_model')
        self.device = config.get('device', torch.device('cpu'))

        assert (24 * 60 * 60) % self.time_intervals == 0, "time_of_day_size should be Int"
        self.time_of_day_size = int((24 * 60 * 60) / self.time_intervals)
        self.day_of_week_size = 7

        self._logger = getLogger()
        self.times_dim = self.time_series_emb_dim*2 + \
                          self.temp_dim_tid * int(self.if_day_in_week) + self.temp_dim_diw * int(self.if_time_in_day) + self.spatial_emb_dim * int(self.if_spatial)*2
        self.hidden_dim = self.time_series_emb_dim + \
                          self.temp_dim_tid * int(self.if_day_in_week) + self.temp_dim_diw * int(self.if_time_in_day) + self.spatial_emb_dim * int(self.if_spatial)
        self.nodes_mlp_dim = self.num_nodes
        self.length_mlp_dim =self.times_dim
        if self.if_time_in_day:
            self.time_in_day_emb = nn.Parameter(torch.empty(self.time_of_day_size, self.temp_dim_tid))
            nn.init.kaiming_uniform_(self.time_in_day_emb, a=0, mode='fan_in')

        if self.if_day_in_week:
            self.day_in_week_emb = nn.Parameter(torch.empty(self.day_of_week_size, self.temp_dim_diw))
            nn.init.xavier_uniform_(self.day_in_week_emb)
        if self.if_spatial:
            self.node_emb = nn.Parameter(torch.empty(self.num_nodes, self.spatial_emb_dim))
            nn.init.xavier_uniform_(self.node_emb)
        self.stem = nn.Conv2d(in_channels=self.input_window, out_channels=self.hidden_dim, kernel_size=(1, 1),bias=True)  # Specify in_channels
        
        self.mixer_blocks = nn.ModuleList([MixerBlock(self.nodes_mlp_dim, self.times_dim) for _ in range(self.num_blocks)])

        self.pre_head_layer_norm = nn.LayerNorm(normalized_shape=self.nodes_mlp_dim)  # Specify the normalized_shape for layer normalization
        self.head = nn.Linear(320, self.output_window)
        self.node_embedding = nn.Linear(self.num_nodes, self.spatial_emb_dim)
        self.activation = nn.ReLU()
        # self.act = nn.ReLU(inplace=True)
        self.supports = [torch.tensor(i).to(self.device) for i in self.adj_mx]
        self.decompsition = series_decomp()
        # self.W_P1 = nn.ModuleList()
        # self.W_P2 = nn.ModuleList()
        # for _ in range(self.input_window//self.patch_len): 
        #     self.W_P1.append(nn.Linear(self.patch_len, self.d_model))
        #     self.W_P2.append(nn.Linear(self.d_model, self.d_model))
        # self.head_linear = nn.Linear(self.input_window//self.patch_len * self.d_model, self.input_window)
    def forward(self, batch):
        
        input_data = batch['X']  # [B, L, N, C]
        time_series = input_data[..., :1]
        adj_emb = self.node_embedding(self.supports[0])
        if self.if_spatial:
            spatial_embedding = self.node_emb
        else:
            spatial_embedding = None
        if self.if_time_in_day:
            tid_data = input_data[..., 1]
            time_in_day_emb = self.time_in_day_emb[(tid_data[:, -1, :] * self.time_of_day_size).type(torch.LongTensor)]
        else:
            time_in_day_emb = None
        if self.if_day_in_week:
            diw_data = torch.argmax(input_data[..., 2:], dim=-1)
            day_in_week_emb = self.day_in_week_emb[(diw_data[:, -1, :]).type(torch.LongTensor)]
        else:
            day_in_week_emb = None
        x = self.stem(time_series)
        x = x.view(x.size(0), x.size(1), -1)  # Rearrange for tokenization
        trend_time_series,season_time_series = self.decompsition(x)
        batch_size, _, num_nodes, _ = time_series.shape
        node_emb = []
        if self.if_spatial:
            node_emb.append(spatial_embedding.unsqueeze(0).expand(batch_size, -1, -1).transpose(1, 2))
            node_emb.append(adj_emb.unsqueeze(0).expand(batch_size, -1, -1).transpose(1, 2))
        tem_emb = []
        if time_in_day_emb is not None:
            tem_emb.append(time_in_day_emb.transpose(1, 2))
        if day_in_week_emb is not None:
            tem_emb.append(day_in_week_emb.transpose(1, 2))
        time_series_emb_trend = self.pre_head_layer_norm(trend_time_series)
        time_series_emb_season = self.pre_head_layer_norm(season_time_series)
        
        hidden_trend = torch.cat([time_series_emb_trend] + tem_emb + node_emb, dim=1)
        hidden_season = torch.cat([time_series_emb_season] + tem_emb + node_emb, dim=1)
        # for block in self.mixer_blocks:
        #     hidden_trend = block(hidden_trend)
        #     hidden_season = block(hidden_season)
        hidden = hidden_trend + hidden_season
        # hidden = torch.einsum('ijkl,kq->ijql', hidden, self.supports[0])
        if self.head:
            # print("hidden: ", hidden.shape)
            hidden = hidden.permute(0,2,1)
            hidden = self.activation(hidden)
            hidden = self.head(hidden)
        
        hidden = hidden.unsqueeze(-1)
        hidden = hidden.permute(0,2,1,3)
        # hidden = hidden.reshape(batch_size,1,num_nodes,self.patch_len,self.input_window//self.patch_len)
        # x_out = []
        # for i in range(self.input_window//self.patch_len):
        #     z = self.W_P1[i](hidden[:,:,:,:,i])
        #     x_out.append(z)
        #     z = self.act(z)
        #     z = self.W_P2[i](z) # ??
        
        # hidden = torch.stack(x_out, dim=4)
       
        # hidden = self.head_linear(hidden.view(hidden.size(0),hidden.size(1),hidden.size(2),-1))
        # hidden = hidden.permute(0,3,2,1)
        # print("hidden",hidden.shape)
        return hidden
    def cal_adj(self, adjtype):
        if adjtype == "scalap":
            self.adj_mx = [calculate_scaled_laplacian(self.adj_mx)]
        elif adjtype == "normlap":
            self.adj_mx = [calculate_normalized_laplacian(self.adj_mx).astype(np.float32).todense()]
        elif adjtype == "symnadj":
            self.adj_mx = [sym_adj(self.adj_mx)]
        elif adjtype == "transition":
            self.adj_mx = [asym_adj(self.adj_mx)]
        elif adjtype == "doubletransition":
            self.adj_mx = [asym_adj(self.adj_mx), asym_adj(np.transpose(self.adj_mx))]
        elif adjtype == "identity":
            self.adj_mx = [np.diag(np.ones(self.adj_mx.shape[0])).astype(np.float32)]
        else:
            assert 0, "adj type not defined"
    def calculate_loss(self, batch):
        y_true = batch['y']
        y_predicted = self.predict(batch)
        y_true = self._scaler.inverse_transform(y_true[..., :self.output_dim])
        y_predicted = self._scaler.inverse_transform(y_predicted[..., :self.output_dim])
        return loss.masked_mae_torch(y_predicted, y_true, 0)

    def predict(self, batch):
        return self.forward(batch)
    
