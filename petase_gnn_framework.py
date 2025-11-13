"""
PETase Thermal Stability Prediction using Graph Neural Networks
================================================================

This framework provides multiple GNN architectures for predicting enzyme
thermal stability from sequence and structure embeddings.

Key Components:
1. Data loading and preprocessing
2. Graph construction from protein structures
3. Multiple GNN architectures (GCN, GAT, GraphSAGE, GIN)
4. Hybrid models combining sequence and structure information
5. Training and evaluation pipeline
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import (
    GCNConv, GATConv, SAGEConv, GINConv,
    global_mean_pool, global_max_pool, global_add_pool,
    BatchNorm, LayerNorm
)
from torch_geometric.data import Data, Dataset, DataLoader
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import pdist, squareform
import warnings
warnings.filterwarnings('ignore')


class ProteinGraphDataset(Dataset):
    """
    Custom dataset for protein graphs with embeddings.
    
    Handles both sequence-based graphs (residue-residue contact) and
    structure-based graphs (spatial distance).
    """
    
    def __init__(self, protein_data, embedding_dir, graph_type='structure',
                 contact_threshold=8.0, transform=None):
        """
        Args:
            protein_data: DataFrame with protein information
            embedding_dir: Path to embedding files
            graph_type: 'structure' or 'sequence'
            contact_threshold: Distance threshold for edges (Angstroms)
        """
        super().__init__(None, transform, None)
        self.protein_data = protein_data
        self.embedding_dir = embedding_dir
        self.graph_type = graph_type
        self.contact_threshold = contact_threshold
        
    def len(self):
        return len(self.protein_data)
    
    def get(self, idx):
        """
        Constructs a graph for a single protein.
        """
        row = self.protein_data.iloc[idx]
        
        # Load embeddings (you'll need to implement based on your file format)
        # This is a placeholder - adapt to your actual embedding format
        node_features = self._load_embeddings(row)
        
        # Construct edges based on graph type
        if self.graph_type == 'structure':
            edge_index, edge_attr = self._construct_structure_graph(row)
        else:
            edge_index, edge_attr = self._construct_sequence_graph(row)
        
        # Target value (Tm)
        y = torch.tensor([row['Tm_numeric']], dtype=torch.float)
        
        # Create PyG Data object
        data = Data(
            x=node_features,
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=y,
            protein_id=row['Name']
        )
        
        return data
    
    def _load_embeddings(self, row):
        """Load pre-computed embeddings for protein."""
        # Placeholder - implement based on your embedding format
        # Could be ESM2/ESM3 sequence embeddings or structure embeddings
        pass
    
    def _construct_structure_graph(self, row):
        """Construct graph based on 3D structure (Ca-Ca distances)."""
        # Placeholder - implement based on PDB or structure embeddings
        pass
    
    def _construct_sequence_graph(self, row):
        """Construct graph based on sequence (k-nearest neighbors)."""
        # Placeholder - implement based on sequence
        pass


# ============================================================================
# GNN ARCHITECTURES
# ============================================================================

class GCN_Model(nn.Module):
    """
    Graph Convolutional Network for protein property prediction.
    
    Simple but effective architecture with:
    - Multiple GCN layers with residual connections
    - Batch normalization
    - Global pooling for graph-level predictions
    """
    
    def __init__(self, input_dim, hidden_dim=256, num_layers=4, 
                 dropout=0.3, pool='mean'):
        super().__init__()
        
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        
        for _ in range(num_layers):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
            self.batch_norms.append(BatchNorm(hidden_dim))
        
        self.dropout = dropout
        self.pool = pool
        
        # Prediction head
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 4, 1)
        )
    
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # Input projection
        x = F.relu(self.input_proj(x))
        
        # GCN layers with residual connections
        for conv, bn in zip(self.convs, self.batch_norms):
            x_new = conv(x, edge_index)
            x_new = bn(x_new)
            x_new = F.relu(x_new)
            x_new = F.dropout(x_new, p=self.dropout, training=self.training)
            x = x + x_new  # Residual connection
        
        # Global pooling
        if self.pool == 'mean':
            x = global_mean_pool(x, batch)
        elif self.pool == 'max':
            x = global_max_pool(x, batch)
        else:  # 'add'
            x = global_add_pool(x, batch)
        
        # Prediction
        out = self.predictor(x)
        return out


class GAT_Model(nn.Module):
    """
    Graph Attention Network with multi-head attention.
    
    Learns to weight edges differently - useful for:
    - Identifying key residue interactions
    - Handling variable connectivity patterns
    """
    
    def __init__(self, input_dim, hidden_dim=256, num_layers=4,
                 num_heads=8, dropout=0.3, pool='mean'):
        super().__init__()
        
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        
        for i in range(num_layers):
            if i == 0:
                self.convs.append(GATConv(hidden_dim, hidden_dim // num_heads,
                                         heads=num_heads, dropout=dropout))
            else:
                self.convs.append(GATConv(hidden_dim, hidden_dim // num_heads,
                                         heads=num_heads, dropout=dropout))
            self.batch_norms.append(BatchNorm(hidden_dim))
        
        self.dropout = dropout
        self.pool = pool
        
        # Prediction head
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        x = F.relu(self.input_proj(x))
        
        for conv, bn in zip(self.convs, self.batch_norms):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Global pooling
        if self.pool == 'mean':
            x = global_mean_pool(x, batch)
        elif self.pool == 'max':
            x = global_max_pool(x, batch)
        else:
            x = global_add_pool(x, batch)
        
        out = self.predictor(x)
        return out


class GraphSAGE_Model(nn.Module):
    """
    GraphSAGE with neighborhood sampling.
    
    Aggregates information from neighboring nodes - good for:
    - Large graphs
    - Learning local structural patterns
    """
    
    def __init__(self, input_dim, hidden_dim=256, num_layers=4,
                 dropout=0.3, aggr='mean'):
        super().__init__()
        
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        
        for _ in range(num_layers):
            self.convs.append(SAGEConv(hidden_dim, hidden_dim, aggr=aggr))
            self.batch_norms.append(BatchNorm(hidden_dim))
        
        self.dropout = dropout
        
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        x = F.relu(self.input_proj(x))
        
        for conv, bn in zip(self.convs, self.batch_norms):
            x_new = conv(x, edge_index)
            x_new = bn(x_new)
            x_new = F.relu(x_new)
            x_new = F.dropout(x_new, p=self.dropout, training=self.training)
            x = x + x_new
        
        x = global_mean_pool(x, batch)
        out = self.predictor(x)
        return out


class GIN_Model(nn.Module):
    """
    Graph Isomorphism Network - maximally expressive GNN.
    
    Distinguishes different graph structures well - useful for:
    - Capturing subtle structural differences
    - Learning complex patterns
    """
    
    def __init__(self, input_dim, hidden_dim=256, num_layers=4, dropout=0.3):
        super().__init__()
        
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        
        for _ in range(num_layers):
            mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 2),
                nn.ReLU(),
                nn.Linear(hidden_dim * 2, hidden_dim)
            )
            self.convs.append(GINConv(mlp, train_eps=True))
            self.batch_norms.append(BatchNorm(hidden_dim))
        
        self.dropout = dropout
        
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        x = F.relu(self.input_proj(x))
        
        for conv, bn in zip(self.convs, self.batch_norms):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        x = global_mean_pool(x, batch)
        out = self.predictor(x)
        return out


class HybridGNN(nn.Module):
    """
    Hybrid model combining sequence and structure information.
    
    Architecture:
    1. Separate GNN branches for sequence and structure graphs
    2. Cross-attention between representations
    3. Fused prediction head
    """
    
    def __init__(self, seq_input_dim, struct_input_dim, hidden_dim=256,
                 num_layers=4, dropout=0.3):
        super().__init__()
        
        # Sequence branch
        self.seq_proj = nn.Linear(seq_input_dim, hidden_dim)
        self.seq_convs = nn.ModuleList([
            GCNConv(hidden_dim, hidden_dim) for _ in range(num_layers)
        ])
        self.seq_norms = nn.ModuleList([
            BatchNorm(hidden_dim) for _ in range(num_layers)
        ])
        
        # Structure branch
        self.struct_proj = nn.Linear(struct_input_dim, hidden_dim)
        self.struct_convs = nn.ModuleList([
            GATConv(hidden_dim, hidden_dim // 8, heads=8, dropout=dropout)
            for _ in range(num_layers)
        ])
        self.struct_norms = nn.ModuleList([
            BatchNorm(hidden_dim) for _ in range(num_layers)
        ])
        
        # Cross-attention
        self.cross_attention = nn.MultiheadAttention(
            hidden_dim, num_heads=8, dropout=dropout, batch_first=True
        )
        
        # Fusion
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Predictor
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        self.dropout = dropout
    
    def forward(self, seq_data, struct_data):
        # Process sequence graph
        x_seq = F.relu(self.seq_proj(seq_data.x))
        for conv, norm in zip(self.seq_convs, self.seq_norms):
            x_seq = conv(x_seq, seq_data.edge_index)
            x_seq = norm(x_seq)
            x_seq = F.relu(x_seq)
        x_seq = global_mean_pool(x_seq, seq_data.batch)
        
        # Process structure graph
        x_struct = F.relu(self.struct_proj(struct_data.x))
        for conv, norm in zip(self.struct_convs, self.struct_norms):
            x_struct = conv(x_struct, struct_data.edge_index)
            x_struct = norm(x_struct)
            x_struct = F.relu(x_struct)
        x_struct = global_mean_pool(x_struct, struct_data.batch)
        
        # Cross-attention
        x_seq_attn, _ = self.cross_attention(
            x_seq.unsqueeze(1), x_struct.unsqueeze(1), x_struct.unsqueeze(1)
        )
        x_seq_attn = x_seq_attn.squeeze(1)
        
        # Fusion
        x_fused = torch.cat([x_seq_attn, x_struct], dim=1)
        x_fused = self.fusion(x_fused)
        
        # Prediction
        out = self.predictor(x_fused)
        return out


# ============================================================================
# TRAINING UTILITIES
# ============================================================================

class EarlyStopping:
    """Early stopping to prevent overfitting."""
    
    def __init__(self, patience=10, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.should_stop = False
    
    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0


def train_epoch(model, loader, optimizer, device, criterion=nn.MSELoss()):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        
        out = model(data)
        loss = criterion(out, data.y)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item() * data.num_graphs
    
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, device, criterion=nn.MSELoss()):
    """Evaluate model."""
    model.eval()
    total_loss = 0
    predictions = []
    targets = []
    
    for data in loader:
        data = data.to(device)
        out = model(data)
        loss = criterion(out, data.y)
        
        total_loss += loss.item() * data.num_graphs
        predictions.extend(out.cpu().numpy())
        targets.extend(data.y.cpu().numpy())
    
    predictions = np.array(predictions)
    targets = np.array(targets)
    
    # Calculate metrics
    mse = total_loss / len(loader.dataset)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(predictions - targets))
    r2 = 1 - (np.sum((targets - predictions) ** 2) / 
              np.sum((targets - np.mean(targets)) ** 2))
    
    return {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'predictions': predictions,
        'targets': targets
    }


def train_model(model, train_loader, val_loader, optimizer, scheduler,
                device, num_epochs=100, early_stopping_patience=15):
    """
    Complete training loop with early stopping.
    """
    criterion = nn.MSELoss()
    early_stopping = EarlyStopping(patience=early_stopping_patience)
    
    best_val_loss = float('inf')
    best_model_state = None
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_rmse': [],
        'val_r2': []
    }
    
    for epoch in range(num_epochs):
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, device, criterion)
        
        # Validate
        val_metrics = evaluate(model, val_loader, device, criterion)
        
        # Update scheduler
        if scheduler is not None:
            scheduler.step(val_metrics['mse'])
        
        # Save history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_metrics['mse'])
        history['val_rmse'].append(val_metrics['rmse'])
        history['val_r2'].append(val_metrics['r2'])
        
        # Save best model
        if val_metrics['mse'] < best_val_loss:
            best_val_loss = val_metrics['mse']
            best_model_state = model.state_dict().copy()
        
        # Print progress
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{num_epochs}")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val Loss: {val_metrics['mse']:.4f}, "
                  f"RMSE: {val_metrics['rmse']:.4f}, "
                  f"R²: {val_metrics['r2']:.4f}")
        
        # Early stopping
        early_stopping(val_metrics['mse'])
        if early_stopping.should_stop:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return model, history


# ============================================================================
# CROSS-VALIDATION AND EVALUATION
# ============================================================================

def cross_validate(model_class, dataset, device, k_folds=5, **model_kwargs):
    """
    Perform k-fold cross-validation.
    """
    kfold = KFold(n_splits=k_folds, shuffle=True, random_state=42)
    
    fold_results = []
    
    for fold, (train_idx, val_idx) in enumerate(kfold.split(dataset)):
        print(f"\n{'='*60}")
        print(f"Fold {fold + 1}/{k_folds}")
        print(f"{'='*60}")
        
        # Create data loaders
        train_subset = [dataset[i] for i in train_idx]
        val_subset = [dataset[i] for i in val_idx]
        
        train_loader = DataLoader(train_subset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=32, shuffle=False)
        
        # Initialize model
        model = model_class(**model_kwargs).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5
        )
        
        # Train
        model, history = train_model(
            model, train_loader, val_loader, optimizer, scheduler,
            device, num_epochs=100, early_stopping_patience=15
        )
        
        # Evaluate on validation set
        val_metrics = evaluate(model, val_loader, device)
        fold_results.append(val_metrics)
        
        print(f"\nFold {fold + 1} Results:")
        print(f"  RMSE: {val_metrics['rmse']:.4f}")
        print(f"  MAE: {val_metrics['mae']:.4f}")
        print(f"  R²: {val_metrics['r2']:.4f}")
    
    # Aggregate results
    avg_rmse = np.mean([r['rmse'] for r in fold_results])
    std_rmse = np.std([r['rmse'] for r in fold_results])
    avg_r2 = np.mean([r['r2'] for r in fold_results])
    std_r2 = np.std([r['r2'] for r in fold_results])
    
    print(f"\n{'='*60}")
    print(f"Cross-Validation Results ({k_folds} folds)")
    print(f"{'='*60}")
    print(f"RMSE: {avg_rmse:.4f} ± {std_rmse:.4f}")
    print(f"R²: {avg_r2:.4f} ± {std_r2:.4f}")
    
    return fold_results


if __name__ == "__main__":
    print("PETase GNN Framework loaded successfully!")
    print("\nAvailable models:")
    print("  - GCN_Model: Graph Convolutional Network")
    print("  - GAT_Model: Graph Attention Network")
    print("  - GraphSAGE_Model: GraphSAGE")
    print("  - GIN_Model: Graph Isomorphism Network")
    print("  - HybridGNN: Hybrid sequence + structure model")
    print("\nNext steps:")
    print("  1. Implement embedding loading in ProteinGraphDataset")
    print("  2. Implement graph construction methods")
    print("  3. Create dataset and run training")
