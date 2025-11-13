"""
Graph Construction for Protein GNNs
====================================

Creates graph representations from protein sequences and structures.
Multiple graph construction strategies are implemented.
"""

import numpy as np
import torch
from torch_geometric.data import Data
from scipy.spatial.distance import cdist
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
import networkx as nx


class ProteinGraphConstructor:
    """
    Constructs protein graphs using various strategies.
    
    Strategies:
    1. Distance-based: Edges between residues within distance threshold
    2. K-nearest neighbors: K nearest residues in sequence/structure space
    3. Sequential: Edges between consecutive residues + long-range contacts
    4. Hybrid: Combination of multiple strategies
    """
    
    def __init__(self, strategy='distance', **kwargs):
        """
        Args:
            strategy: 'distance', 'knn', 'sequential', or 'hybrid'
            **kwargs: Strategy-specific parameters
        """
        self.strategy = strategy
        self.params = kwargs
    
    def construct_from_sequence(self, sequence_embedding, protein_id=None):
        """
        Construct graph from sequence embeddings.
        
        Uses k-NN in embedding space to determine edges.
        """
        # sequence_embedding shape: (num_residues, embedding_dim)
        L, D = sequence_embedding.shape
        
        # Node features
        x = torch.FloatTensor(sequence_embedding)
        
        # Construct edges
        if self.strategy == 'sequential':
            edge_index, edge_attr = self._sequential_edges(L)
        elif self.strategy == 'knn':
            k = self.params.get('k', 10)
            edge_index, edge_attr = self._knn_edges(sequence_embedding, k)
        elif self.strategy == 'hybrid':
            edge_index1, edge_attr1 = self._sequential_edges(L)
            k = self.params.get('k', 5)
            edge_index2, edge_attr2 = self._knn_edges(sequence_embedding, k)
            edge_index, edge_attr = self._merge_edges(
                edge_index1, edge_attr1, edge_index2, edge_attr2
            )
        else:
            raise ValueError(f"Strategy {self.strategy} not supported for sequence")
        
        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    
    def construct_from_structure(self, ca_coords, node_features=None, 
                                  distance_threshold=8.0):
        """
        Construct graph from 3D structure (C-alpha coordinates).
        
        Args:
            ca_coords: (L, 3) array of C-alpha coordinates
            node_features: (L, D) array of node features (default: one-hot position)
            distance_threshold: Distance cutoff for edges (Angstroms)
        """
        L = len(ca_coords)
        
        # Node features (use positional encoding if not provided)
        if node_features is None:
            node_features = self._positional_encoding(L, 64)
        
        x = torch.FloatTensor(node_features)
        
        # Construct edges based on distance
        if self.strategy == 'distance':
            edge_index, edge_attr = self._distance_edges(
                ca_coords, distance_threshold
            )
        elif self.strategy == 'knn':
            k = self.params.get('k', 20)
            edge_index, edge_attr = self._knn_spatial_edges(ca_coords, k)
        elif self.strategy == 'hybrid':
            # Combine sequential + distance-based edges
            edge_index1, edge_attr1 = self._sequential_edges(L)
            edge_index2, edge_attr2 = self._distance_edges(ca_coords, distance_threshold)
            edge_index, edge_attr = self._merge_edges(
                edge_index1, edge_attr1, edge_index2, edge_attr2
            )
        else:
            edge_index, edge_attr = self._distance_edges(ca_coords, distance_threshold)
        
        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    
    def _sequential_edges(self, num_residues):
        """Create edges between consecutive residues."""
        edges = []
        edge_features = []
        
        for i in range(num_residues - 1):
            # Forward edge
            edges.append([i, i + 1])
            edge_features.append([1.0, 0.0])  # [sequential, not_sequential]
            
            # Backward edge (undirected)
            edges.append([i + 1, i])
            edge_features.append([1.0, 0.0])
        
        edge_index = torch.LongTensor(edges).t().contiguous()
        edge_attr = torch.FloatTensor(edge_features)
        
        return edge_index, edge_attr
    
    def _distance_edges(self, coords, threshold=8.0):
        """Create edges between residues within distance threshold."""
        dist_matrix = cdist(coords, coords, metric='euclidean')
        
        # Find pairs within threshold
        i_indices, j_indices = np.where((dist_matrix < threshold) & (dist_matrix > 0))
        
        edges = np.stack([i_indices, j_indices], axis=0)
        
        # Edge features: [distance, normalized_distance]
        distances = dist_matrix[i_indices, j_indices]
        edge_features = np.stack([
            distances,
            distances / threshold
        ], axis=1)
        
        edge_index = torch.LongTensor(edges)
        edge_attr = torch.FloatTensor(edge_features)
        
        return edge_index, edge_attr
    
    def _knn_edges(self, embeddings, k=10):
        """Create k-nearest neighbor edges in embedding space."""
        from sklearn.neighbors import kneighbors_graph
        
        # Build k-NN graph
        knn_graph = kneighbors_graph(
            embeddings, k, mode='distance', include_self=False
        )
        
        # Convert to edge list
        knn_graph = knn_graph.tocoo()
        edges = np.stack([knn_graph.row, knn_graph.col], axis=0)
        
        # Edge features: similarity in embedding space
        distances = knn_graph.data
        edge_features = np.stack([
            distances,
            1.0 / (1.0 + distances)  # Similarity score
        ], axis=1)
        
        edge_index = torch.LongTensor(edges)
        edge_attr = torch.FloatTensor(edge_features)
        
        return edge_index, edge_attr
    
    def _knn_spatial_edges(self, coords, k=20):
        """Create k-nearest neighbor edges in 3D space."""
        from sklearn.neighbors import kneighbors_graph
        
        knn_graph = kneighbors_graph(
            coords, k, mode='distance', include_self=False
        )
        
        knn_graph = knn_graph.tocoo()
        edges = np.stack([knn_graph.row, knn_graph.col], axis=0)
        
        distances = knn_graph.data
        edge_features = np.stack([
            distances,
            np.exp(-distances / 4.0)  # RBF kernel
        ], axis=1)
        
        edge_index = torch.LongTensor(edges)
        edge_attr = torch.FloatTensor(edge_features)
        
        return edge_index, edge_attr
    
    def _merge_edges(self, edge_index1, edge_attr1, edge_index2, edge_attr2):
        """Merge two sets of edges, removing duplicates."""
        # Concatenate
        edge_index = torch.cat([edge_index1, edge_index2], dim=1)
        edge_attr = torch.cat([edge_attr1, edge_attr2], dim=0)
        
        # Remove duplicates (keep first occurrence)
        edge_dict = {}
        unique_edges = []
        unique_attrs = []
        
        for i in range(edge_index.shape[1]):
            edge = tuple(edge_index[:, i].tolist())
            if edge not in edge_dict:
                edge_dict[edge] = True
                unique_edges.append(edge_index[:, i])
                unique_attrs.append(edge_attr[i])
        
        edge_index = torch.stack(unique_edges, dim=1)
        edge_attr = torch.stack(unique_attrs, dim=0)
        
        return edge_index, edge_attr
    
    def _positional_encoding(self, seq_len, d_model=64):
        """
        Sinusoidal positional encoding for residue positions.
        
        Helps the model understand sequence order when using structure graphs.
        """
        position = np.arange(seq_len)[:, np.newaxis]
        div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
        
        pe = np.zeros((seq_len, d_model))
        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)
        
        return pe
    
    def add_edge_types(self, data, ca_coords=None):
        """
        Add edge type annotations:
        - Sequential (i, i+1)
        - Short-range (|i-j| <= 5)
        - Medium-range (5 < |i-j| <= 12)
        - Long-range (|i-j| > 12)
        """
        edge_index = data.edge_index.numpy()
        edge_types = []
        
        for i in range(edge_index.shape[1]):
            src, dst = edge_index[0, i], edge_index[1, i]
            seq_dist = abs(src - dst)
            
            if seq_dist == 1:
                edge_types.append(0)  # Sequential
            elif seq_dist <= 5:
                edge_types.append(1)  # Short-range
            elif seq_dist <= 12:
                edge_types.append(2)  # Medium-range
            else:
                edge_types.append(3)  # Long-range
        
        data.edge_type = torch.LongTensor(edge_types)
        return data


class MultiScaleGraphConstructor:
    """
    Constructs multi-scale protein graphs.
    
    Creates multiple graph representations at different scales:
    - Fine: All residues, detailed connectivity
    - Coarse: Clustered residues, high-level structure
    """
    
    def __init__(self, coarse_grain_factor=5):
        self.coarse_grain_factor = coarse_grain_factor
    
    def construct_hierarchical(self, embeddings, ca_coords=None):
        """
        Construct both fine and coarse-grained graphs.
        
        Returns:
            fine_graph: Detailed residue-level graph
            coarse_graph: Coarse-grained graph
            hierarchy_edges: Connections between levels
        """
        # Fine-grained graph (all residues)
        fine_constructor = ProteinGraphConstructor(strategy='hybrid', k=10)
        
        if ca_coords is not None:
            fine_graph = fine_constructor.construct_from_structure(
                ca_coords, embeddings
            )
        else:
            fine_graph = fine_constructor.construct_from_sequence(embeddings)
        
        # Coarse-grain by clustering consecutive residues
        num_residues = embeddings.shape[0]
        cluster_size = self.coarse_grain_factor
        num_clusters = (num_residues + cluster_size - 1) // cluster_size
        
        # Aggregate features for each cluster
        coarse_features = []
        cluster_assignments = []
        
        for i in range(num_clusters):
            start_idx = i * cluster_size
            end_idx = min((i + 1) * cluster_size, num_residues)
            
            # Mean-pool features within cluster
            cluster_feat = embeddings[start_idx:end_idx].mean(axis=0)
            coarse_features.append(cluster_feat)
            
            # Track which residues belong to this cluster
            for j in range(start_idx, end_idx):
                cluster_assignments.append(i)
        
        coarse_features = np.array(coarse_features)
        
        # Construct coarse graph (edges between nearby clusters)
        coarse_edges = []
        for i in range(num_clusters - 1):
            coarse_edges.append([i, i + 1])
            coarse_edges.append([i + 1, i])
        
        coarse_graph = Data(
            x=torch.FloatTensor(coarse_features),
            edge_index=torch.LongTensor(coarse_edges).t().contiguous()
        )
        
        # Hierarchy edges (connecting fine to coarse level)
        hierarchy_edges = torch.LongTensor([
            list(range(num_residues)),
            cluster_assignments
        ])
        
        return fine_graph, coarse_graph, hierarchy_edges


def visualize_protein_graph(data, title="Protein Graph", output_path=None):
    """Visualize protein graph structure."""
    import matplotlib.pyplot as plt
    
    # Convert to NetworkX for visualization
    G = nx.Graph()
    edge_index = data.edge_index.numpy()
    
    for i in range(edge_index.shape[1]):
        src, dst = edge_index[0, i], edge_index[1, i]
        G.add_edge(src, dst)
    
    plt.figure(figsize=(12, 8))
    pos = nx.spring_layout(G, k=0.5, iterations=50)
    
    nx.draw(G, pos,
            node_color='lightblue',
            node_size=50,
            edge_color='gray',
            alpha=0.6,
            width=0.5)
    
    plt.title(f"{title}\nNodes: {data.x.shape[0]}, Edges: {edge_index.shape[1]}")
    plt.axis('off')
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"✓ Graph visualization saved to {output_path}")
    else:
        plt.show()
    
    plt.close()


if __name__ == "__main__":
    print("Graph Constructor Module")
    print("="*60)
    print("\nAvailable strategies:")
    print("  - distance: Edges between residues within distance threshold")
    print("  - knn: K-nearest neighbors in embedding/spatial space")
    print("  - sequential: Consecutive residues with optional long-range")
    print("  - hybrid: Combination of sequential + distance/knn")
    print("\nExample:")
    print("  constructor = ProteinGraphConstructor(strategy='hybrid', k=10)")
    print("  graph = constructor.construct_from_structure(ca_coords)")
