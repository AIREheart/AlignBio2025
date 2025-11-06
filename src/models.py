import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    def __init__(self, in_channels=21, hidden=128):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, hidden, 5, padding=2)
        self.conv2 = nn.Conv1d(hidden, hidden, 3, padding=1)
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = F.gelu(self.conv1(x))
        x = F.gelu(self.conv2(x))
        return self.pool(x).squeeze(-1)

class FusionModel(nn.Module):
    def __init__(self, in_channels=21, glob_dim=128, n_tasks=1):
        super().__init__()
        self.cnn = SimpleCNN(in_channels, hidden=128)
        self.gnn_lin1 = nn.Linear(in_channels, 128)
        self.gnn_lin2 = nn.Linear(128, 128)
        self.glob = nn.Sequential(nn.Linear(glob_dim, 256), nn.GELU(), nn.Linear(256, 64))
        self.fuse = nn.Sequential(nn.Linear(128*2+64, 256), nn.GELU(), nn.Linear(256, 128))
        self.out = nn.Linear(128, n_tasks)

    def forward(self, seq_onehot, adj, glob_feat):
        cnn_h = self.cnn(seq_onehot)
        h = F.gelu(self.gnn_lin1(seq_onehot))
        h2 = torch.bmm(adj, h)
        gnn_h = F.gelu(self.gnn_lin2(h2)).mean(1)
        glob_h = self.glob(glob_feat)
        hcat = torch.cat([cnn_h, gnn_h, glob_h], dim=-1)
        fused = self.fuse(hcat)
        return self.out(fused)
