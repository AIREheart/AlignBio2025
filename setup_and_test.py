"""
Setup Verification and Quick Test Script
=========================================

Verifies installation and runs a quick test with dummy data.
"""

import sys
import subprocess

def check_dependencies():
    """Check if required packages are installed."""
    print("Checking dependencies...")
    print("="*60)
    
    required = {
        'torch': 'PyTorch',
        'torch_geometric': 'PyTorch Geometric',
        'pandas': 'Pandas',
        'numpy': 'NumPy',
        'sklearn': 'Scikit-learn',
        'matplotlib': 'Matplotlib',
        'scipy': 'SciPy'
    }
    
    missing = []
    
    for package, name in required.items():
        try:
            __import__(package)
            print(f"✓ {name}")
        except ImportError:
            print(f"✗ {name} - NOT INSTALLED")
            missing.append(name)
    
    if missing:
        print(f"\n⚠️  Missing packages: {', '.join(missing)}")
        print("Please install with: pip install -r requirements.txt")
        return False
    
    print("\n✓ All required packages installed!")
    return True


def check_cuda():
    """Check CUDA availability."""
    try:
        import torch
        print("\n" + "="*60)
        print("GPU Check")
        print("="*60)
        
        if torch.cuda.is_available():
            print(f"✓ CUDA available: {torch.cuda.get_device_name(0)}")
            print(f"  CUDA version: {torch.version.cuda}")
            print(f"  Number of GPUs: {torch.cuda.device_count()}")
        else:
            print("ℹ️  CUDA not available - will use CPU")
            print("  (Training will be slower but still work)")
    except Exception as e:
        print(f"Error checking CUDA: {e}")


def run_quick_test():
    """Run a quick test with small dummy dataset."""
    print("\n" + "="*60)
    print("Running Quick Test")
    print("="*60)
    
    try:
        import torch
        import numpy as np
        from torch_geometric.data import Data
        from petase_gnn_framework import GCN_Model
        
        print("\nCreating dummy graph...")
        
        # Create a small dummy protein graph
        num_nodes = 50  # 50 residues
        node_dim = 128  # Simplified embedding
        
        x = torch.randn(num_nodes, node_dim)  # Node features
        
        # Create simple sequential edges
        edge_list = []
        for i in range(num_nodes - 1):
            edge_list.append([i, i+1])
            edge_list.append([i+1, i])
        
        # Add some random long-range edges
        for _ in range(20):
            i, j = np.random.choice(num_nodes, 2, replace=False)
            edge_list.append([i, j])
            edge_list.append([j, i])
        
        edge_index = torch.LongTensor(edge_list).t().contiguous()
        y = torch.FloatTensor([[60.0]])  # Dummy Tm value
        
        data = Data(x=x, edge_index=edge_index, y=y)
        
        print(f"  Nodes: {data.x.shape[0]}")
        print(f"  Edges: {data.edge_index.shape[1]}")
        print(f"  Node feature dim: {data.x.shape[1]}")
        
        # Test model
        print("\nInitializing GCN model...")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = GCN_Model(
            input_dim=node_dim,
            hidden_dim=64,
            num_layers=3
        ).to(device)
        
        print(f"  Device: {device}")
        print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        # Test forward pass
        print("\nTesting forward pass...")
        data = data.to(device)
        data.batch = torch.zeros(num_nodes, dtype=torch.long).to(device)
        
        model.eval()
        with torch.no_grad():
            output = model(data)
        
        print(f"  Input Tm: {y.item():.2f}°C")
        print(f"  Output shape: {output.shape}")
        print(f"  Predicted Tm (untrained): {output.item():.2f}°C")
        
        print("\n✓ Quick test passed! Framework is working correctly.")
        return True
        
    except Exception as e:
        print(f"\n✗ Quick test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def print_next_steps():
    """Print guidance on next steps."""
    print("\n" + "="*60)
    print("NEXT STEPS")
    print("="*60)
    
    print("""
1. GENERATE EMBEDDINGS (Option A - Real):
   
   from embedding_generator import generate_all_embeddings_from_dataset
   import pandas as pd
   
   df = pd.read_excel('MASTER_DB.xlsx', sheet_name='150petase')
   embeddings = generate_all_embeddings_from_dataset(df)
   
   Note: Requires ESM2 installed: pip install fair-esm
   
2. GENERATE EMBEDDINGS (Option B - Quick test with dummy data):
   
   python run_petase_gnn.py
   
   This will automatically create dummy embeddings for testing.

3. CUSTOMIZE:
   
   - Edit graph construction strategy in run_petase_gnn.py
   - Try different GNN architectures
   - Tune hyperparameters
   - Add your own features

4. ANALYZE RESULTS:
   
   Check the 'results/' folder for:
   - model_comparison.png
   - training_curves.png
   - prediction_scatter.png

5. FOR PRODUCTION:
   
   - Replace dummy embeddings with real ESM2/ESM3 embeddings
   - Use cross-validation for robust evaluation
   - Implement ensemble models
   - Add uncertainty quantification
    """)


def main():
    """Main setup and test function."""
    print("\n" + "="*60)
    print("PETASE GNN FRAMEWORK - SETUP & TEST")
    print("="*60)
    
    # Check dependencies
    if not check_dependencies():
        print("\n⚠️  Please install missing dependencies first.")
        return
    
    # Check CUDA
    check_cuda()
    
    # Run quick test
    test_passed = run_quick_test()
    
    if test_passed:
        print_next_steps()
    else:
        print("\n⚠️  Setup test failed. Please check the error messages above.")
        print("   Try reinstalling dependencies or contact support.")


if __name__ == "__main__":
    main()
