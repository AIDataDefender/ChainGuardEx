class GINEStateEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int = 128,
        hidden_channels: int = 64,
        out_channels: int = 96,
        num_layers: int = 3,
        dropout: float = 0.5,
        train_eps: bool = False,
        pooling: str = "mean",
        edge_attr_dim: int = 32,  # dimension of your concatenated edge features
    ):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        self.pooling = pooling

        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()

        # Build GINE layers
        # First layer: in_channels → hidden
        self.convs.append(
            GINEConv(
                nn=nn.Sequential(
                    nn.Linear(in_channels, hidden_channels),
                    nn.ReLU(),
                    nn.Linear(hidden_channels, hidden_channels),
                ),
                eps=0.0,
                train_eps=train_eps,
                edge_dim=edge_attr_dim,
            )
        )
        self.batch_norms.append(nn.BatchNorm1d(hidden_channels))

        # Intermediate layers
        for _ in range(num_layers - 2):
            self.convs.append(
                GINEConv(
                    nn=nn.Sequential(
                        nn.Linear(hidden_channels, hidden_channels),
                        nn.ReLU(),
                        nn.Linear(hidden_channels, hidden_channels),
                    ),
                    eps=0.0,
                    train_eps=train_eps,
                    edge_dim=edge_attr_dim,
                )
            )
            self.batch_norms.append(nn.BatchNorm1d(hidden_channels))

        # Final layer
        if num_layers > 1:
            self.convs.append(
                GINEConv(
                    nn=nn.Sequential(
                        nn.Linear(hidden_channels, out_channels),
                        nn.ReLU(),
                        nn.Linear(out_channels, out_channels),
                    ),
                    eps=0.0,
                    train_eps=train_eps,
                    edge_dim=edge_attr_dim,
                )
            )
            self.batch_norms.append(nn.BatchNorm1d(out_channels))

        # In case of single-layer config
        self.final_projection = None
        if num_layers == 1:
            self.final_projection = nn.Linear(hidden_channels, out_channels)

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        batch = getattr(
            data, "batch", torch.zeros(x.size(0), device=x.device, dtype=torch.long)
        )

        # Pass through GINEConv layers
        for i, (conv, bn) in enumerate(zip(self.convs, self.batch_norms)):
            x = conv(x, edge_index, edge_attr)
            x = bn(x)
            x = F.relu(x)
            if i < len(self.convs) - 1:
                x = F.dropout(x, p=self.dropout, training=self.training)

        if self.final_projection is not None:
            x = self.final_projection(x)

        # Graph-level pooling
        if self.pooling == "mean":
            out = global_mean_pool(x, batch)
        else:
            out = global_add_pool(x, batch)
        return out
