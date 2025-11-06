"""
Model definitions: SimpleCNN, SimpleGNN, FusionWithEmb
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    def __init__(self, in_channels=21, channels=128):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, channels, kernel_size=5, padding=2)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool1d(1)
    def forward(self, x):
        x = x.transpose(1,2)
        x = F.gelu(self.conv1(x))
        x = F.gelu(self.conv2(x))
        return self.pool(x).squeeze(-1)

class FusionWithEmb(nn.Module):
    def __init__(self, in_channels=21, glob_dim=64, n_tasks=1):
        super().__init__()
        self.cnn = SimpleCNN(in_channels=in_channels, channels=128)
        self.gnn_lin1 = nn.Linear(in_channels,128); self.gnn_lin2 = nn.Linear(128,128)
        self.glob_lin = nn.Sequential(nn.Linear(glob_dim,256), nn.GELU(), nn.Linear(256,64))
        self.fuse = nn.Sequential(nn.Linear(128+128+64,256), nn.GELU(), nn.Dropout(0.2), nn.Linear(256,128))
        self.heads = nn.ModuleList([nn.Sequential(nn.Linear(128,64), nn.GELU(), nn.Linear(64,1)) for _ in range(n_tasks)])
    def forward(self, seq_onehot, adj, glob):
        cnn_h = self.cnn(seq_onehot)
        h = F.gelu(self.gnn_lin1(seq_onehot))
        h2 = torch.bmm(adj, h); gnn_h = F.gelu(self.gnn_lin2(h2))
        mask = seq_onehot.sum(-1) > 0; mask = mask.float().unsqueeze(-1)
        gnn_h = (gnn_h*mask).sum(1)/(mask.sum(1)+1e-6)
        glob_h = self.glob_lin(glob)
        hcat = torch.cat([cnn_h, gnn_h, glob_h], dim=-1)
        h = self.fuse(hcat)
        outs = [head(h).squeeze(-1) for head in self.heads]
        return torch.stack(outs, dim=1)
