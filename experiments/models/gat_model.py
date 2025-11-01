import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl.nn.pytorch import HeteroGraphConv, GATConv
from typing import Dict, Tuple, Optional, List
import logging
import traceback

from experiments.utils.logger import setup_logger

os.makedirs("Logs", exist_ok=True)
logger = setup_logger("Logs/gat_model.log")


class HeteroGATLayer(nn.Module):
    """
    Heterogeneous Graph Attention Network layer for processing multi-type graphs.
    Applies GAT convolution for each edge type in the heterograph.
    """
    
    def __init__(self, in_feat_dict: Dict[str, int], out_feat: int, num_heads: int = 8, dropout: float = 0.1):
        """
        Args:
            in_feat_dict: Dictionary mapping node types to input feature dimensions
            out_feat: Output feature dimension for all node types
            num_heads: Number of attention heads
            dropout: Dropout rate (feat_drop and attn_drop in GATConv)
        """
        super(HeteroGATLayer, self).__init__()
        self.in_feat_dict = in_feat_dict
        self.out_feat = out_feat
        self.num_heads = num_heads
        
        # Define GAT for each edge type
        edge_type_dict = {}
        for src_type in in_feat_dict.keys():
            for dst_type in in_feat_dict.keys():
                # Add all possible edge type combinations that might exist
                # Note: GATConv uses feat_drop and attn_drop instead of dropout
                edge_type_dict[(src_type, 'to', dst_type)] = GATConv(
                    in_feat_dict[src_type], out_feat, num_heads, 
                    feat_drop=dropout, attn_drop=dropout
                )
        
        self.hetero_conv = HeteroGraphConv(edge_type_dict, aggregate='mean')
    
    def forward(self, g: dgl.DGLGraph, input_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        output_dict = {}
        for node_type in input_dict.keys():
            output_dict[node_type] = input_dict[node_type]
        
        # Apply convolutions only for edge types that exist in the graph
        for src_type, rel, dst_type in g.canonical_etypes:
            try:
                if (src_type, 'to', dst_type) in self.hetero_conv.mods:
                    src_feat = input_dict.get(src_type, None)
                    if src_feat is not None and src_type in output_dict:
                        pass
            except:
                pass
        
        return self.hetero_conv(g, input_dict)


class GraphEncoder(nn.Module):
    """
    Multi-layer Graph Attention Network encoder for heterogeneous graphs.
    Encodes graph structure into fixed-size embeddings via pooling.
    """
    
    def __init__(self, in_feat_dict: Dict[str, int], hidden_dim: int = 128, num_layers: int = 2, 
                num_heads: int = 8, dropout: float = 0.1):
        """
        Args:
            in_feat_dict: Input feature dimensions for each node type
            hidden_dim: Hidden dimension for GAT layers
            num_layers: Number of GAT layers
            num_heads: Number of attention heads
            dropout: Dropout rate
        """
        super(GraphEncoder, self).__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.node_types = list(in_feat_dict.keys())
        
        # Create input projection layers for each node type
        self.input_proj = nn.ModuleDict()
        for node_type, feat_dim in in_feat_dict.items():
            if feat_dim != hidden_dim:
                self.input_proj[node_type] = nn.Linear(feat_dim, hidden_dim)
            else:
                self.input_proj[node_type] = nn.Identity()
        
        # Create GAT layers
        self.gat_layers = nn.ModuleList()
        for _ in range(num_layers):
            layer_in_feat = {node_type: hidden_dim for node_type in in_feat_dict.keys()}
            self.gat_layers.append(HeteroGATLayer(layer_in_feat, hidden_dim, num_heads, dropout))
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, g: dgl.DGLGraph) -> Dict[str, torch.Tensor]:
        """
        Args:
            g: DGL heterograph with node features in g.nodes[node_type].data['feat']
        
        Returns:
            Dictionary of encoded node features for each node type
        """
        # Project input features
        h_dict = {}
        for node_type in self.node_types:
            if node_type in g.nodes:
                feat = g.nodes[node_type].data['feat']
                h_dict[node_type] = self.input_proj[node_type](feat)
        
        # Apply GAT layers
        for gat_layer in self.gat_layers:
            h_dict = gat_layer(g, h_dict)
            # Flatten multi-head attention output if needed
            for node_type in h_dict:
                if len(h_dict[node_type].shape) == 3:  # (num_nodes, num_heads, out_feat)
                    h_dict[node_type] = h_dict[node_type].mean(dim=1)  # Average heads
                h_dict[node_type] = self.dropout(F.relu(h_dict[node_type]))
        
        return h_dict
    
    def graph_pooling(self, h_dict: Dict[str, torch.Tensor], pooling_type: str = 'mean') -> torch.Tensor:
        """
        Pool node embeddings to get graph-level embedding.
        
        Args:
            h_dict: Dictionary of node embeddings
            pooling_type: 'mean', 'max', or 'sum'
        
        Returns:
            Graph-level embedding tensor
        """
        pooled = []
        device = None
        
        for node_type in sorted(h_dict.keys()):
            if h_dict[node_type].shape[0] > 0:  # Only if nodes exist
                if device is None:
                    device = h_dict[node_type].device
                
                if pooling_type == 'mean':
                    pooled.append(h_dict[node_type].mean(dim=0))
                elif pooling_type == 'max':
                    pooled.append(h_dict[node_type].max(dim=0)[0])
                elif pooling_type == 'sum':
                    pooled.append(h_dict[node_type].sum(dim=0))
                else:
                    pooled.append(h_dict[node_type].mean(dim=0))
        
        if pooled:
            return torch.cat(pooled, dim=0)
        else:
            if device is None:
                device = torch.device('cpu')
            return torch.zeros(len(h_dict) * self.hidden_dim, device=device)


class ChainGuardV2Model(nn.Module):
    """
    Dual-input model combining Graph Attention Networks and CodeBERT embeddings
    for smart contract vulnerability detection.
    
    Architecture:
    - Graph Encoder: HeteroGAT processes the heterogeneous contract graph
    - Code Encoder: Linear projection of CodeBERT embeddings
    - Fusion: Combines graph and code representations
    - Multi-task Classifier: Predicts vulnerabilities from different analysis types
      * Main vulnerability labels
      * Call Graph (CG) vulnerabilities  
      * Control Flow Graph (CFG) vulnerabilities
    """
    
    def __init__(self, code_dim: int = 768, hidden_dim: int = 256, 
                    num_classes: int = 1, graph_hidden_dim: int = 128, 
                    num_gat_layers: int = 2, num_heads: int = 8, 
                    dropout: float = 0.1, in_feat_dict: Optional[Dict[str, int]] = None,
                    use_multi_task: bool = True):
        """
        Args:
            code_dim: Dimension of CodeBERT embeddings (usually 768)
            hidden_dim: Hidden dimension for fusion and classification layers
            num_classes: Number of vulnerability classes (or 1 for multi-label)
            graph_hidden_dim: Hidden dimension for GAT layers
            num_gat_layers: Number of GAT layers
            num_heads: Number of attention heads in GAT
            dropout: Dropout rate
            in_feat_dict:   Input feature dimensions for graph node types
                            If None, uses default dimensions
            use_multi_task: Whether to use multi-task learning (CG + CFG + main labels)
        """
        super(ChainGuardV2Model, self).__init__()
        
        # Default feature dimensions
        if in_feat_dict is None:
            in_feat_dict = {
                'function': 128,
                'block': 128,
                'ast_node': 64,
                'dfg_node': 32,
            }
        #'ast_node': 9225, 'block': 653, 'dfg_node': 0, 'function': 30}
        self.code_dim = code_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.use_multi_task = use_multi_task
        
        # Graph encoder (GAT)
        self.graph_encoder = GraphEncoder(
            in_feat_dict=in_feat_dict,
            hidden_dim=graph_hidden_dim,
            num_layers=num_gat_layers,
            num_heads=num_heads,
            dropout=dropout
        )
        
        # Calculate graph embedding dimension
        self.graph_embedding_dim = len(in_feat_dict) * graph_hidden_dim
        
        # Code encoder
        self.code_encoder = nn.Sequential(
            nn.Linear(code_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(self.graph_embedding_dim + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Main classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )
        
        # Multi-task learning heads (if enabled)
        if self.use_multi_task:
            self.cg_classifier = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, num_classes)
            )
            self.cfg_classifier = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, num_classes)
            )
    
    def forward(self, graph: dgl.DGLGraph, code_features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the model.
        
        Args:
            graph: DGL heterograph with node features
            code_features: Code embeddings from CodeBERT 
                            Shape: (batch_size, code_dim) if pooled at batch level
                            Shape: (num_samples, code_dim) if not yet pooled
        
        Returns:
            Main vulnerability predictions (batch_size, num_classes)
            Or dict with multi-task outputs if use_multi_task=True
        """
        try:
            # Handle code features shape
            if len(code_features.shape) == 3:
                # If (batch_size, seq_len, code_dim), pool to (batch_size, code_dim)
                code_features = code_features.mean(dim=1)
            
            batch_size = code_features.shape[0]
            
            # Encode graph
            h_dict = self.graph_encoder(graph)
            
            # Pool graph to get graph-level embeddings
            # For batched graphs, we need to get per-sample embeddings
            graph_embeddings_list = []
            
            # Get node batch assignment from batched graph
            batch_num_nodes = graph.batch_num_nodes()
            node_offset = 0
            
            for sample_idx in range(len(batch_num_nodes)):
                num_nodes = batch_num_nodes[sample_idx].item()
                
                # Extract embeddings for this sample's nodes
                sample_h_dict = {}
                for node_type in h_dict.keys():
                    if node_type in graph.ntypes:
                        # Get node indices for this sample
                        node_type_nodes = graph.nodes(node_type)
                        # For this sample
                        sample_h_dict[node_type] = h_dict[node_type]
                
                # Pool this sample's graph embedding
                pooled = []
                for node_type in sorted(sample_h_dict.keys()):
                    if sample_h_dict[node_type].shape[0] > 0:
                        pooled.append(sample_h_dict[node_type].mean(dim=0))
                
                if pooled:
                    graph_embeddings_list.append(torch.cat(pooled, dim=0))
                else:
                    graph_embeddings_list.append(
                        torch.zeros(self.graph_embedding_dim, device=code_features.device)
                    )
                
                node_offset += num_nodes
            
            # Stack graph embeddings
            if graph_embeddings_list:
                graph_embedding = torch.stack(graph_embeddings_list, dim=0)
            else:
                graph_embedding = torch.zeros(batch_size, self.graph_embedding_dim, device=code_features.device)
            
            # Ensure same batch size
            if graph_embedding.shape[0] != batch_size:
                # Pad or truncate to match batch size
                if graph_embedding.shape[0] < batch_size:
                    pad_size = batch_size - graph_embedding.shape[0]
                    graph_embedding = torch.cat([
                        graph_embedding,
                        torch.zeros(pad_size, self.graph_embedding_dim, device=code_features.device)
                    ], dim=0)
                else:
                    graph_embedding = graph_embedding[:batch_size]
            
            # Encode code
            code_embedding = self.code_encoder(code_features)
            
            # Fuse graph and code embeddings
            fused = torch.cat([graph_embedding, code_embedding], dim=-1)
            fused_embedding = self.fusion(fused)
            
            # Classify
            predictions = self.classifier(fused_embedding)
            
            if self.use_multi_task:
                # Return dict with all outputs for multi-task learning
                return {
                    'main': predictions,
                    'cg': self.cg_classifier(fused_embedding),
                    'cfg': self.cfg_classifier(fused_embedding)
                }
            else:
                return predictions
        
        except Exception as e:
            logger.error(f"Error in forward pass: {e}")
            logger.error(traceback.format_exc())
            # Return zero predictions on error
            batch_size = code_features.shape[0]
            if self.use_multi_task:
                return {
                    'main': torch.zeros(batch_size, self.num_classes, device=code_features.device),
                    'cg': torch.zeros(batch_size, self.num_classes, device=code_features.device),
                    'cfg': torch.zeros(batch_size, self.num_classes, device=code_features.device)
                }
            else:
                return torch.zeros(batch_size, self.num_classes, device=code_features.device)


def train_epoch(model, train_loader, optimizer, criterion, device, use_multi_task=True):
    """
    Train for one epoch with support for multi-task learning.
    
    Args:
        model: VulnerabilityDetectionModel instance
        train_loader: Iterator yielding batch dicts with keys:
                     {'graph', 'code', 'labels', 'cg_labels', 'cfg_labels', ...}
        optimizer: PyTorch optimizer
        criterion: Loss function
        device: torch device
        use_multi_task: Whether model uses multi-task learning
    
    Returns:
        Dictionary with loss metrics: {'total': float, 'main': float, 'cg': float, 'cfg': float}
    """
    model.train()
    total_loss = 0.0
    main_loss = 0.0
    cg_loss = 0.0
    cfg_loss = 0.0
    num_batches = 0
    
    for batch_data in train_loader:
        try:
            # Extract batch components
            batch_graph = batch_data['graph']
            batch_code = batch_data['code']
            batch_labels = batch_data['labels']
            batch_cg_labels = batch_data.get('cg_labels', None)
            batch_cfg_labels = batch_data.get('cfg_labels', None)
            
            # Skip if critical data missing
            if batch_graph is None or batch_code is None or batch_labels is None:
                logger.warning(f"Skipping batch with missing data")
                continue
            
            # Move data to device
            batch_graph = batch_graph.to(device)
            batch_code = batch_code.to(device)
            batch_labels = batch_labels.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            predictions = model(batch_graph, batch_code)
            
            # Compute loss
            if use_multi_task and isinstance(predictions, dict):
                # Main task loss
                loss_main = criterion(predictions['main'], batch_labels)
                total_batch_loss = loss_main
                main_loss += loss_main.item()
                
                # CG task loss (if available)
                if batch_cg_labels is not None:
                    batch_cg_labels = batch_cg_labels.to(device)
                    loss_cg = criterion(predictions['cg'], batch_cg_labels)
                    total_batch_loss = total_batch_loss + 0.5 * loss_cg  # Weight auxiliary tasks
                    cg_loss += loss_cg.item()
                
                # CFG task loss (if available)
                if batch_cfg_labels is not None:
                    batch_cfg_labels = batch_cfg_labels.to(device)
                    loss_cfg = criterion(predictions['cfg'], batch_cfg_labels)
                    total_batch_loss = total_batch_loss + 0.5 * loss_cfg
                    cfg_loss += loss_cfg.item()
            else:
                # Single-task loss
                total_batch_loss = criterion(predictions, batch_labels)
                main_loss += total_batch_loss.item()
            
            # Backward pass
            total_batch_loss.backward()
            optimizer.step()
            
            total_loss += total_batch_loss.item()
            num_batches += 1
        
        except Exception as e:
            logger.error(f"Error in training batch: {e}")
            logger.error(traceback.format_exc())
            continue
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    avg_main_loss = main_loss / num_batches if num_batches > 0 else 0.0
    avg_cg_loss = cg_loss / num_batches if num_batches > 0 else 0.0
    avg_cfg_loss = cfg_loss / num_batches if num_batches > 0 else 0.0
    
    return {
        'total': avg_loss,
        'main': avg_main_loss,
        'cg': avg_cg_loss,
        'cfg': avg_cfg_loss
    }


def evaluate(model, val_loader, criterion, device, use_multi_task=True):
    """
    Evaluate model on validation set with multi-task support.
    
    Args:
        model: VulnerabilityDetectionModel instance
        val_loader: Iterator yielding batch dicts with keys:
                   {'graph', 'code', 'labels', 'cg_labels', 'cfg_labels', ...}
        criterion: Loss function
        device: torch device
        use_multi_task: Whether model uses multi-task learning
    
    Returns:
        Dictionary with metrics: {
            'loss': float,
            'main_loss': float, 
            'cg_loss': float,
            'cfg_loss': float,
            'predictions': tensor,
            'labels': tensor
        }
    """
    model.eval()
    total_loss = 0.0
    main_loss = 0.0
    cg_loss = 0.0
    cfg_loss = 0.0
    all_predictions = []
    all_labels = []
    num_batches = 0
    
    with torch.no_grad():
        for batch_data in val_loader:
            try:
                # Extract batch components
                batch_graph = batch_data['graph']
                batch_code = batch_data['code']
                batch_labels = batch_data['labels']
                batch_cg_labels = batch_data.get('cg_labels', None)
                batch_cfg_labels = batch_data.get('cfg_labels', None)
                
                # Skip if critical data missing
                if batch_graph is None or batch_code is None or batch_labels is None:
                    continue
                
                # Move data to device
                batch_graph = batch_graph.to(device)
                batch_code = batch_code.to(device)
                batch_labels = batch_labels.to(device)
                
                # Forward pass
                predictions = model(batch_graph, batch_code)
                
                # Compute loss
                if use_multi_task and isinstance(predictions, dict):
                    # Main task loss
                    loss_main = criterion(predictions['main'], batch_labels)
                    total_batch_loss = loss_main
                    main_loss += loss_main.item()
                    
                    # CG task loss (if available)
                    if batch_cg_labels is not None:
                        batch_cg_labels = batch_cg_labels.to(device)
                        loss_cg = criterion(predictions['cg'], batch_cg_labels)
                        total_batch_loss = total_batch_loss + 0.5 * loss_cg
                        cg_loss += loss_cg.item()
                    
                    # CFG task loss (if available)
                    if batch_cfg_labels is not None:
                        batch_cfg_labels = batch_cfg_labels.to(device)
                        loss_cfg = criterion(predictions['cfg'], batch_cfg_labels)
                        total_batch_loss = total_batch_loss + 0.5 * loss_cfg
                        cfg_loss += loss_cfg.item()
                    
                    all_predictions.append(predictions['main'].cpu())
                else:
                    total_batch_loss = criterion(predictions, batch_labels)
                    main_loss += total_batch_loss.item()
                    all_predictions.append(predictions.cpu())
                
                total_loss += total_batch_loss.item()
                all_labels.append(batch_labels.cpu())
                num_batches += 1
            
            except Exception as e:
                logger.error(f"Error in evaluation batch: {e}")
                logger.error(traceback.format_exc())
                continue
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    avg_main_loss = main_loss / num_batches if num_batches > 0 else 0.0
    avg_cg_loss = cg_loss / num_batches if num_batches > 0 else 0.0
    avg_cfg_loss = cfg_loss / num_batches if num_batches > 0 else 0.0
    all_predictions = torch.cat(all_predictions, dim=0) if all_predictions else torch.tensor([])
    all_labels = torch.cat(all_labels, dim=0) if all_labels else torch.tensor([])
    
    return {
        'loss': avg_loss,
        'main_loss': avg_main_loss,
        'cg_loss': avg_cg_loss,
        'cfg_loss': avg_cfg_loss,
        'predictions': all_predictions,
        'labels': all_labels
    }


if __name__ == "__main__":
    # Example initialization
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    in_feat_dict = {
        'function': 128,
        'block': 128,
        'ast_node': 64,
        'dfg_node': 32,
    }
    
    model = ChainGuardV2Model(
        code_dim=768,
        hidden_dim=256,
        num_classes=1,
        graph_hidden_dim=128,
        num_gat_layers=2,
        num_heads=8,
        dropout=0.1,
        in_feat_dict=in_feat_dict,
        use_multi_task=True  # Enable multi-task learning
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Model initialized successfully!")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Device: {device}")
    print(f"Multi-task learning enabled: {model.use_multi_task}")
    print(f"\nModel architecture:")
    print(f"  - Graph encoder: {model.graph_embedding_dim} → {model.hidden_dim}")
    print(f"  - Code encoder: {model.code_dim} → {model.hidden_dim}")
    print(f"  - Fusion: {model.graph_embedding_dim + model.hidden_dim} → {model.hidden_dim}")
    print(f"  - Main classifier: {model.hidden_dim} → {model.num_classes}")
    if model.use_multi_task:
        print(f"  - CG classifier: {model.hidden_dim} → {model.num_classes}")
        print(f"  - CFG classifier: {model.hidden_dim} → {model.num_classes}")
