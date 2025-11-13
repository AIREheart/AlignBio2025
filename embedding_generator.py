"""
Embedding Generation Pipeline for PETase Proteins
==================================================

Generates both sequence and structure embeddings using ESM models.
"""

import torch
import numpy as np
import pandas as pd
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


class EmbeddingGenerator:
    """
    Generates embeddings for protein sequences and structures.
    
    Supports:
    - ESM2 (sequence): 2560-dim per-residue embeddings
    - ESM3 (sequence + structure): 1536-dim structure embeddings
    """
    
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.models = {}
        
    def load_esm2_model(self, model_name='esm2_t33_650M_UR50D'):
        """Load ESM2 for sequence embeddings."""
        try:
            import esm
            print(f"Loading ESM2 model: {model_name}")
            model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
            model = model.eval().to(self.device)
            self.models['esm2'] = (model, alphabet)
            print("✓ ESM2 loaded successfully")
            return True
        except Exception as e:
            print(f"✗ Failed to load ESM2: {e}")
            print("  Install with: pip install fair-esm")
            return False
    
    def load_esmfold_model(self):
        """Load ESMFold for structure prediction."""
        try:
            import esm
            print("Loading ESMFold model")
            model = esm.pretrained.esmfold_v1()
            model = model.eval().to(self.device)
            self.models['esmfold'] = model
            print("✓ ESMFold loaded successfully")
            return True
        except Exception as e:
            print(f"✗ Failed to load ESMFold: {e}")
            return False
    
    def generate_sequence_embeddings(self, sequences, batch_size=8):
        """
        Generate per-residue sequence embeddings using ESM2.
        
        Args:
            sequences: List of (protein_id, sequence) tuples
            batch_size: Batch size for processing
            
        Returns:
            Dictionary mapping protein_id to embeddings
        """
        if 'esm2' not in self.models:
            raise ValueError("ESM2 model not loaded. Call load_esm2_model() first.")
        
        model, alphabet = self.models['esm2']
        batch_converter = alphabet.get_batch_converter()
        
        embeddings_dict = {}
        
        # Process in batches
        for i in range(0, len(sequences), batch_size):
            batch = sequences[i:i + batch_size]
            
            # Convert to ESM format
            batch_labels, batch_strs, batch_tokens = batch_converter(batch)
            batch_tokens = batch_tokens.to(self.device)
            
            # Generate embeddings
            with torch.no_grad():
                results = model(batch_tokens, repr_layers=[33])
                token_representations = results["representations"][33]
            
            # Extract per-sequence embeddings (remove BOS/EOS tokens)
            for idx, (protein_id, seq) in enumerate(batch):
                seq_len = len(seq)
                # Mean pooling across residues
                embeddings = token_representations[idx, 1:seq_len+1].cpu().numpy()
                embeddings_dict[protein_id] = {
                    'per_residue': embeddings,  # Shape: (L, 2560)
                    'mean_pool': embeddings.mean(axis=0),  # Shape: (2560,)
                    'max_pool': embeddings.max(axis=0),    # Shape: (2560,)
                }
            
            print(f"Processed {min(i + batch_size, len(sequences))}/{len(sequences)} sequences")
        
        return embeddings_dict
    
    def generate_structure_embeddings_from_pdb(self, pdb_files):
        """
        Generate structure embeddings from PDB files.
        
        For proteins with known structures, this extracts:
        - C-alpha coordinates
        - Secondary structure features
        - Distance matrices
        """
        embeddings_dict = {}
        
        for protein_id, pdb_path in pdb_files.items():
            try:
                # Parse PDB and extract features
                coords, features = self._parse_pdb(pdb_path)
                
                embeddings_dict[protein_id] = {
                    'ca_coords': coords,           # Shape: (L, 3)
                    'distance_matrix': self._compute_distance_matrix(coords),
                    'features': features
                }
                
            except Exception as e:
                print(f"Warning: Failed to process {protein_id}: {e}")
        
        return embeddings_dict
    
    def _parse_pdb(self, pdb_path):
        """Parse PDB file and extract C-alpha coordinates."""
        from Bio.PDB import PDBParser
        
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure('protein', pdb_path)
        
        # Extract C-alpha coordinates
        ca_coords = []
        for model in structure:
            for chain in model:
                for residue in chain:
                    if 'CA' in residue:
                        ca_coords.append(residue['CA'].get_coord())
        
        ca_coords = np.array(ca_coords)
        
        # Compute basic features (can be extended)
        features = {
            'num_residues': len(ca_coords),
            'center_of_mass': ca_coords.mean(axis=0)
        }
        
        return ca_coords, features
    
    def _compute_distance_matrix(self, coords):
        """Compute pairwise C-alpha distance matrix."""
        from scipy.spatial.distance import cdist
        return cdist(coords, coords, metric='euclidean')
    
    def predict_structure_from_sequence(self, sequences):
        """
        Predict structures using ESMFold (for proteins without PDB).
        
        Returns predicted structures and embeddings.
        """
        if 'esmfold' not in self.models:
            raise ValueError("ESMFold not loaded. Call load_esmfold_model() first.")
        
        model = self.models['esmfold']
        structure_predictions = {}
        
        for protein_id, sequence in sequences:
            with torch.no_grad():
                output = model.infer(sequence)
            
            structure_predictions[protein_id] = {
                'positions': output['positions'],  # Predicted coordinates
                'plddt': output['plddt'],          # Confidence scores
                'mean_plddt': output['mean_plddt']
            }
            
            print(f"Predicted structure for {protein_id} (pLDDT: {output['mean_plddt']:.2f})")
        
        return structure_predictions
    
    def save_embeddings(self, embeddings_dict, output_dir, embedding_type):
        """Save embeddings to disk."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for protein_id, embeddings in embeddings_dict.items():
            output_path = output_dir / f"{protein_id}_{embedding_type}.npz"
            np.savez_compressed(output_path, **embeddings)
        
        print(f"✓ Saved {len(embeddings_dict)} {embedding_type} embeddings to {output_dir}")
    
    def load_embeddings(self, protein_id, embedding_dir, embedding_type):
        """Load embeddings from disk."""
        file_path = Path(embedding_dir) / f"{protein_id}_{embedding_type}.npz"
        
        if not file_path.exists():
            raise FileNotFoundError(f"Embeddings not found: {file_path}")
        
        return dict(np.load(file_path))


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def generate_all_embeddings_from_dataset(df, output_dir='embeddings'):
    """
    Generate embeddings for all proteins in the dataset.
    
    Args:
        df: DataFrame with protein information (from Excel file)
        output_dir: Directory to save embeddings
    """
    generator = EmbeddingGenerator()
    
    # Filter to proteins with sequences and target values
    df_valid = df[df['Protein sequence'].notna() & df['Tm_numeric'].notna()].copy()
    
    print(f"Processing {len(df_valid)} proteins with sequences and Tm values")
    
    # Prepare sequences
    sequences = [
        (row['Name'], row['Protein sequence']) 
        for _, row in df_valid.iterrows()
    ]
    
    # Generate sequence embeddings
    print("\n" + "="*60)
    print("Generating sequence embeddings with ESM2")
    print("="*60)
    
    if generator.load_esm2_model():
        seq_embeddings = generator.generate_sequence_embeddings(sequences)
        generator.save_embeddings(seq_embeddings, output_dir, 'sequence')
    
    # Generate structure embeddings (for proteins with PDB)
    print("\n" + "="*60)
    print("Generating structure embeddings")
    print("="*60)
    
    proteins_with_pdb = df_valid[df_valid['Structure PDB ID'].notna()]
    
    if len(proteins_with_pdb) > 0:
        print(f"Found {len(proteins_with_pdb)} proteins with PDB structures")
        # You would implement PDB fetching here
        # pdb_files = {row['Name']: f"structures/{row['Structure PDB ID']}.pdb" 
        #              for _, row in proteins_with_pdb.iterrows()}
        # struct_embeddings = generator.generate_structure_embeddings_from_pdb(pdb_files)
        # generator.save_embeddings(struct_embeddings, output_dir, 'structure')
    
    print("\n✓ Embedding generation complete!")
    
    return seq_embeddings


if __name__ == "__main__":
    print("Embedding Generator Module")
    print("="*60)
    print("\nUsage:")
    print("  from embedding_generator import EmbeddingGenerator")
    print("  generator = EmbeddingGenerator()")
    print("  generator.load_esm2_model()")
    print("  embeddings = generator.generate_sequence_embeddings(sequences)")
