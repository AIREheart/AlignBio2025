"""
Fusion CNN + (simple) GNN model.
Supports using per-residue embeddings (as additional node features) and
global pooled embeddings (appended to global features).
"""
import torch, torch.nn as nn, torch.nn.functional as F

class SequenceCNN(nn.Module):
    def __init__(self, in_channels=21, hidden=256):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, hidden, kernel_size=5, padding=2)
        self.conv2 = nn.Conv1d(hidden, hidden, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.norm = nn.LayerNorm(hidden)
    def forward(self, x):
        # x: [B, L, C]
        x = x.transpose(1,2)  # [B, C, L]
        x = F.gelu(self.conv1(x))
        x = F.gelu(self.conv2(x))
        x = self.pool(x).squeeze(-1)  # [B, hidden]
        x = self.norm(x)
        return x

class SimpleGNN(nn.Module):
    def __init__(self, node_in=21, hidden=128, n_layers=3):
        super().__init__()
        self.lin_in = nn.Linear(node_in, hidden)
        self.layers = nn.ModuleList([nn.Linear(hidden, hidden) for _ in range(n_layers)])
        self.act = nn.GELU()
    def forward(self, node_feats, adj):
        # node_feats: [B, L, F], adj: [B, L, L]
        h = self.lin_in(node_feats)  # [B, L, hidden]
        for lin in self.layers:
            m = torch.bmm(adj, h)  # simple message sum
            h = self.act(lin(m))
        # global pool over residues
        mask = (node_feats.sum(-1) != 0).float().unsqueeze(-1)  # [B, L, 1]
        h_sum = (h * mask).sum(1)  # [B, hidden]
        denom = mask.sum(1).clamp(min=1.0)
        h_pool = h_sum / denom
        return h_pool

class FusionModel(nn.Module):
    def __init__(self, seq_channels=21, node_feat_dim=21, global_dim=128, out_tasks=3):
        super().__init__()
        self.seq_cnn = SequenceCNN(in_channels=seq_channels, hidden=256)
        self.gnn = SimpleGNN(node_in=node_feat_dim, hidden=128)
        self.glob_mlp = nn.Sequential(nn.Linear(global_dim, 256), nn.GELU(), nn.Linear(256, 128))
        self.fuse = nn.Sequential(nn.Linear(256+128+128, 256), nn.GELU(), nn.Dropout(0.2))
        self.heads = nn.ModuleList([nn.Sequential(nn.Linear(256,128), nn.GELU(), nn.Linear(128,1)) for _ in range(out_tasks)])
    def forward(self, seq_onehot, node_feats, adj, glob_vec):
        # seq_onehot: [B,L,C_seq], node_feats: [B,L,F_node], adj: [B,L,L], glob_vec: [B, global_dim]
        seq_h = self.seq_cnn(seq_onehot)           # [B, 256]
        gnn_h = self.gnn(node_feats, adj)          # [B, 128]
        glob_h = self.glob_mlp(glob_vec)           # [B, 128]
        cat = torch.cat([seq_h, gnn_h, glob_h], dim=-1)  # [B, 512]
        fused = self.fuse(cat)                     # [B, 256]
        outs = [h(fused).squeeze(-1) for h in self.heads]  # list length out_tasks
        return torch.stack(outs, dim=1)            # [B, out_tasks]
