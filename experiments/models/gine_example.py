"""
Example usage of graph batching with GINE model for variable-sized heterogeneous graphs
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch_geometric.nn import GINEConv, global_mean_pool
import torch.nn.functional as F

from experiments.dataset import CustomDataset
from experiments.utils.graph_converter import collate_hetero_graphs
from experiments.utils.logger import setup_logger

logger = setup_logger("Logs/gine_example.log")


class SimpleGINEModel(nn.Module):
    """
    Simple GINE model that handles batched heterogeneous graphs
    converted to homogeneous format.
    """
    def __init__(
        self,
        node_feat_dim: int = 128,
        edge_feat_dim: int = 32,
        hidden_dim: int = 64,
        output_dim: int = 10,
        num_layers: int = 3,
        dropout: float = 0.5
    ):
        super().__init__()
        
        self.num_layers = num_layers
        self.dropout = dropout
        
        # First layer
        self.conv1 = GINEConv(
            nn=nn.Sequential(
                nn.Linear(node_feat_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            ),
            edge_dim=edge_feat_dim
        )
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        
        # Middle layers
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        for _ in range(num_layers - 2):
            self.convs.append(
                GINEConv(
                    nn=nn.Sequential(
                        nn.Linear(hidden_dim, hidden_dim),
                        nn.ReLU(),
                        nn.Linear(hidden_dim, hidden_dim)
                    ),
                    edge_dim=edge_feat_dim
                )
            )
            self.bns.append(nn.BatchNorm1d(hidden_dim))
        
        # Final layer
        self.conv_final = GINEConv(
            nn=nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            ),
            edge_dim=edge_feat_dim
        )
        self.bn_final = nn.BatchNorm1d(hidden_dim)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim)
        )
    
    def forward(self, data):
        """
        Forward pass
        
        Args:
            data: PyTorch Geometric Batch object with:
                - x: Node features [total_nodes_in_batch, node_feat_dim]
                - edge_index: Edge connectivity [2, total_edges_in_batch]
                - edge_attr: Edge features [total_edges_in_batch, edge_feat_dim]
                - batch: Batch assignment vector [total_nodes_in_batch]
        
        Returns:
            Graph-level predictions [batch_size, output_dim]
        """
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        batch = data.batch
        
        # First conv layer
        x = self.conv1(x, edge_index, edge_attr)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Middle layers
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index, edge_attr)
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Final layer
        x = self.conv_final(x, edge_index, edge_attr)
        x = self.bn_final(x)
        x = F.relu(x)
        
        # Global pooling: aggregate node features to graph-level
        # This handles variable graph sizes automatically
        x = global_mean_pool(x, batch)  # [batch_size, hidden_dim]
        
        # Classification
        out = self.classifier(x)  # [batch_size, output_dim]
        
        return out


def train_one_epoch(model, dataloader, optimizer, criterion, device):
    """
    Train for one epoch
    """
    model.train()
    total_loss = 0
    num_batches = 0
    
    for batch_data in dataloader:
        if batch_data is None:
            continue
        
        # Get graph batch (already on device from collate_fn)
        graph_batch = batch_data['graph']
        labels = batch_data['labels'].to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(graph_batch)
        
        # Compute loss
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
    
    return total_loss / max(num_batches, 1)


def main():
    """
    Example training loop with variable-sized heterogeneous graphs
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Load dataset
    logger.info("Loading dataset...")
    dataset = CustomDataset(
        tokenizer=None,
        args=None,
        source="DAppSCAN",
        force_reload=False,
        load_type="both"
    )
    logger.info(f"Dataset loaded with {len(dataset)} samples")
    
    # Create DataLoader with custom collate function
    # This handles variable-sized graphs automatically
    dataloader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        collate_fn=collate_hetero_graphs,  # KEY: Use custom collate
        num_workers=0  # Set to 0 for debugging, increase for performance
    )
    
    # Initialize model
    # Note: You may need to adjust dimensions based on your actual data
    model = SimpleGINEModel(
        node_feat_dim=128,  # Adjust based on your node features
        edge_feat_dim=32,   # Adjust based on your edge features
        hidden_dim=64,
        output_dim=10,      # Number of vulnerability classes
        num_layers=3,
        dropout=0.5
    ).to(device)
    
    # Optimizer and loss
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    # Training loop
    num_epochs = 5
    logger.info(f"Starting training for {num_epochs} epochs...")
    
    for epoch in range(num_epochs):
        avg_loss = train_one_epoch(model, dataloader, optimizer, criterion, device)
        logger.info(f"Epoch {epoch+1}/{num_epochs} - Average Loss: {avg_loss:.4f}")
    
    logger.info("Training complete!")


if __name__ == "__main__":
    main()
