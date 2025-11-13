"""
Complete Pipeline for PETase Thermal Stability Prediction
==========================================================

End-to-end pipeline that:
1. Loads data from Excel
2. Generates/loads embeddings
3. Constructs protein graphs
4. Trains GNN models
5. Evaluates and compares different architectures
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch_geometric.data import Data, Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Import our modules
from petase_gnn_framework import (
    GCN_Model, GAT_Model, GraphSAGE_Model, GIN_Model,
    train_model, evaluate, cross_validate
)
from graph_constructor import ProteinGraphConstructor


class PETaseDataset(Dataset):
    """
    Custom dataset for PETase proteins.
    
    Handles loading of embeddings and graph construction.
    """
    
    def __init__(self, dataframe, embedding_dir, graph_strategy='hybrid',
                 use_structure=False, transform=None):
        super().__init__(None, transform, None)
        
        self.df = dataframe.reset_index(drop=True)
        self.embedding_dir = Path(embedding_dir)
        self.graph_constructor = ProteinGraphConstructor(
            strategy=graph_strategy, k=10
        )
        self.use_structure = use_structure
        
        # Store valid indices (proteins with embeddings and Tm values)
        self.valid_indices = self._get_valid_indices()
        
        print(f"Dataset initialized: {len(self.valid_indices)} valid samples")
    
    def _get_valid_indices(self):
        """Get indices of proteins with valid data."""
        valid = []
        for idx in range(len(self.df)):
            row = self.df.iloc[idx]
            if pd.notna(row.get('Tm_numeric')) and self._has_embeddings(row):
                valid.append(idx)
        return valid
    
    def _has_embeddings(self, row):
        """Check if embeddings exist for this protein."""
        protein_name = row['Name']
        seq_file = self.embedding_dir / f"{protein_name}_sequence.npz"
        return seq_file.exists()
    
    def len(self):
        return len(self.valid_indices)
    
    def get(self, idx):
        """Load and construct graph for a single protein."""
        actual_idx = self.valid_indices[idx]
        row = self.df.iloc[actual_idx]
        
        # Load embeddings
        protein_name = row['Name']
        embeddings = self._load_embeddings(protein_name)
        
        # Construct graph
        if self.use_structure and 'ca_coords' in embeddings:
            graph = self.graph_constructor.construct_from_structure(
                embeddings['ca_coords'],
                embeddings['per_residue']
            )
        else:
            graph = self.graph_constructor.construct_from_sequence(
                embeddings['per_residue']
            )
        
        # Add target value (Tm)
        graph.y = torch.FloatTensor([row['Tm_numeric']])
        
        # Add metadata
        graph.protein_id = protein_name
        graph.protein_family = row['Protein']
        
        return graph
    
    def _load_embeddings(self, protein_name):
        """Load embeddings from disk."""
        seq_file = self.embedding_dir / f"{protein_name}_sequence.npz"
        
        if not seq_file.exists():
            raise FileNotFoundError(f"Embeddings not found: {seq_file}")
        
        data = dict(np.load(seq_file))
        return data


def create_dummy_embeddings(df, output_dir='embeddings'):
    """
    Create dummy embeddings for testing (REPLACE WITH REAL EMBEDDINGS).
    
    This simulates ESM2 embeddings with random data.
    In production, use the EmbeddingGenerator to create real embeddings.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Creating dummy embeddings (for testing only)...")
    
    for idx, row in df.iterrows():
        if pd.notna(row.get('Tm_numeric')):
            protein_name = row['Name']
            
            # Simulate sequence embeddings (2560-dim like ESM2)
            seq_length = 300  # Approximate PETase length
            per_residue = np.random.randn(seq_length, 2560).astype(np.float32)
            mean_pool = per_residue.mean(axis=0)
            max_pool = per_residue.max(axis=0)
            
            # Save
            output_file = output_dir / f"{protein_name}_sequence.npz"
            np.savez_compressed(
                output_file,
                per_residue=per_residue,
                mean_pool=mean_pool,
                max_pool=max_pool
            )
    
    print(f"✓ Created dummy embeddings for {len(df)} proteins in {output_dir}")


def load_and_prepare_data(excel_path, embedding_dir='embeddings'):
    """
    Load PETase dataset and prepare for training.
    """
    print("="*70)
    print("LOADING DATA")
    print("="*70)
    
    # Load Excel file
    df = pd.read_excel(excel_path, sheet_name='150petase')
    
    # Convert Tm to numeric (handle mixed types)
    df['Tm_numeric'] = pd.to_numeric(df['Tm'], errors='coerce')
    
    # Filter to valid samples
    df_valid = df[df['Tm_numeric'].notna()].copy()
    
    print(f"Total proteins: {len(df)}")
    print(f"Proteins with Tm values: {len(df_valid)}")
    print(f"Tm range: {df_valid['Tm_numeric'].min():.1f} - {df_valid['Tm_numeric'].max():.1f}°C")
    print(f"Mean Tm: {df_valid['Tm_numeric'].mean():.1f} ± {df_valid['Tm_numeric'].std():.1f}°C")
    
    # Check if embeddings exist, create dummy if not
    embedding_dir = Path(embedding_dir)
    if not embedding_dir.exists() or not any(embedding_dir.glob('*_sequence.npz')):
        print("\n⚠️  No embeddings found. Creating dummy embeddings for testing.")
        print("    Replace with real embeddings using EmbeddingGenerator!")
        create_dummy_embeddings(df_valid, embedding_dir)
    
    return df_valid


def train_and_evaluate_model(model_class, model_name, dataset, device,
                             input_dim=2560, hidden_dim=256, num_layers=4):
    """
    Train and evaluate a single model.
    """
    print("\n" + "="*70)
    print(f"TRAINING {model_name}")
    print("="*70)
    
    # Split data
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    
    # Initialize model
    model = model_class(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=0.3
    ).to(device)
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {num_params:,}")
    
    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-4,
        weight_decay=1e-5
    )
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    
    # Train
    model, history = train_model(
        model, train_loader, val_loader, optimizer, scheduler,
        device, num_epochs=100, early_stopping_patience=15
    )
    
    # Final evaluation
    val_metrics = evaluate(model, val_loader, device)
    
    print(f"\nFinal Results for {model_name}:")
    print(f"  RMSE: {val_metrics['rmse']:.3f}°C")
    print(f"  MAE: {val_metrics['mae']:.3f}°C")
    print(f"  R²: {val_metrics['r2']:.3f}")
    
    return model, history, val_metrics


def compare_models(dataset, device, input_dim=2560):
    """
    Train and compare multiple GNN architectures.
    """
    models_to_test = [
        (GCN_Model, "GCN"),
        (GAT_Model, "GAT"),
        (GraphSAGE_Model, "GraphSAGE"),
        (GIN_Model, "GIN"),
    ]
    
    results = {}
    
    for model_class, model_name in models_to_test:
        try:
            model, history, metrics = train_and_evaluate_model(
                model_class, model_name, dataset, device, input_dim
            )
            
            results[model_name] = {
                'model': model,
                'history': history,
                'metrics': metrics
            }
        except Exception as e:
            print(f"\n✗ Failed to train {model_name}: {e}")
    
    return results


def plot_results(results, output_dir='results'):
    """
    Create visualizations of results.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Model comparison barplot
    model_names = list(results.keys())
    rmse_values = [results[m]['metrics']['rmse'] for m in model_names]
    r2_values = [results[m]['metrics']['r2'] for m in model_names]
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # RMSE comparison
    axes[0].bar(model_names, rmse_values, color='skyblue', edgecolor='navy')
    axes[0].set_ylabel('RMSE (°C)', fontsize=12)
    axes[0].set_title('Model Comparison: RMSE', fontsize=14, fontweight='bold')
    axes[0].grid(axis='y', alpha=0.3)
    
    # R² comparison
    axes[1].bar(model_names, r2_values, color='lightcoral', edgecolor='darkred')
    axes[1].set_ylabel('R²', fontsize=12)
    axes[1].set_title('Model Comparison: R²', fontsize=14, fontweight='bold')
    axes[1].grid(axis='y', alpha=0.3)
    axes[1].set_ylim([0, 1])
    
    plt.tight_layout()
    plt.savefig(output_dir / 'model_comparison.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved model comparison to {output_dir / 'model_comparison.png'}")
    plt.close()
    
    # 2. Training curves for best model
    best_model_name = max(results.keys(), key=lambda m: results[m]['metrics']['r2'])
    history = results[best_model_name]['history']
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss curves
    axes[0].plot(history['train_loss'], label='Train', linewidth=2)
    axes[0].plot(history['val_loss'], label='Validation', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('MSE Loss', fontsize=12)
    axes[0].set_title(f'{best_model_name}: Training Progress', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(alpha=0.3)
    
    # R² curve
    axes[1].plot(history['val_r2'], color='green', linewidth=2)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('R²', fontsize=12)
    axes[1].set_title(f'{best_model_name}: Validation R²', fontsize=14, fontweight='bold')
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'training_curves.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved training curves to {output_dir / 'training_curves.png'}")
    plt.close()
    
    # 3. Prediction scatter plot
    metrics = results[best_model_name]['metrics']
    predictions = metrics['predictions'].flatten()
    targets = metrics['targets'].flatten()
    
    fig, ax = plt.subplots(figsize=(8, 8))
    
    ax.scatter(targets, predictions, alpha=0.6, s=100, edgecolors='k', linewidth=0.5)
    
    # Perfect prediction line
    min_val, max_val = targets.min(), targets.max()
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
    
    # Add statistics
    slope, intercept, r_value, p_value, std_err = stats.linregress(targets, predictions)
    ax.plot(targets, slope * targets + intercept, 'b-', linewidth=2, alpha=0.7, label='Linear Fit')
    
    ax.set_xlabel('Experimental Tm (°C)', fontsize=14)
    ax.set_ylabel('Predicted Tm (°C)', fontsize=14)
    ax.set_title(f'{best_model_name}: Predicted vs Experimental Tm\n' +
                f'R² = {metrics["r2"]:.3f}, RMSE = {metrics["rmse"]:.3f}°C',
                fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'prediction_scatter.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved prediction scatter to {output_dir / 'prediction_scatter.png'}")
    plt.close()


def main():
    """
    Main execution pipeline.
    """
    print("\n" + "="*70)
    print("PETASE THERMAL STABILITY PREDICTION WITH GNNS")
    print("="*70)
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    # Load data
    df = load_and_prepare_data(
        './MASTER_DB.xlsx',
        embedding_dir='embeddings'
    )
    
    # Create dataset
    print("\n" + "="*70)
    print("CREATING DATASET")
    print("="*70)
    
    dataset = PETaseDataset(
        df,
        embedding_dir='embeddings',
        graph_strategy='hybrid',
        use_structure=False
    )
    
    print(f"Dataset size: {len(dataset)} proteins")
    
    # Check a sample graph
    sample = dataset[0]
    print(f"\nSample graph:")
    print(f"  Nodes: {sample.x.shape[0]} (dim: {sample.x.shape[1]})")
    print(f"  Edges: {sample.edge_index.shape[1]}")
    print(f"  Target Tm: {sample.y.item():.2f}°C")
    print(f"  Protein: {sample.protein_id}")
    
    # Train and compare models
    results = compare_models(dataset, device, input_dim=sample.x.shape[1])
    
    # Plot results
    if results:
        print("\n" + "="*70)
        print("GENERATING VISUALIZATIONS")
        print("="*70)
        plot_results(results)
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    if results:
        print("\nModel Performance:")
        for model_name, result in sorted(
            results.items(),
            key=lambda x: x[1]['metrics']['r2'],
            reverse=True
        ):
            metrics = result['metrics']
            print(f"\n{model_name}:")
            print(f"  RMSE: {metrics['rmse']:.3f}°C")
            print(f"  MAE:  {metrics['mae']:.3f}°C")
            print(f"  R²:   {metrics['r2']:.3f}")
        
        best_model = max(results.keys(), key=lambda m: results[m]['metrics']['r2'])
        print(f"\n🏆 Best model: {best_model}")
    
    print("\n" + "="*70)
    print("PIPELINE COMPLETE!")
    print("="*70)


if __name__ == "__main__":
    main()
