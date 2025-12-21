import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.function as fn
from .baseFrameModel import BaseModel




class EdgeGatedHeteroGINLayer(nn.Module):
    def __init__(
        self,
        hidden_dim,
        edge_dim,
        rel_names,
        dropout=0.1
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.rel_names = rel_names

        # Edge encoder → scalar gate
        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_dim, edge_dim // 2),
            nn.ReLU(),
            nn.Linear(edge_dim // 2, 1),
        )

        # Relation embeddings (lightweight)
        self.rel_emb = nn.ParameterDict({
            "_".join(r): nn.Parameter(torch.randn(16)) for r in rel_names
        })

        # Type-specific GIN MLPs (applied after aggregation)
        self.node_mlp = nn.ModuleDict({
            "ast_node": nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ),
            "cfg_node": nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ),
        })

        # Learnable ε per node type
        self.eps = nn.ParameterDict({
            "ast_node": nn.Parameter(torch.zeros(1)),
            "cfg_node": nn.Parameter(torch.zeros(1)),
        })

    def forward(self, g, h_dict, edge_feat_dict):
        with g.local_scope():
            for ntype in h_dict:
                g.nodes[ntype].data["h"] = h_dict[ntype]

            for rel in self.rel_names:
                if g.num_edges(rel) == 0:
                    continue

                src, etype, dst = rel
                e_feat = edge_feat_dict[rel]

                # Relation embedding injection
                rel_emb = self.rel_emb["_".join(rel)].unsqueeze(0).expand(
                    e_feat.size(0), -1
                )

                # Edge gate (scalar, bounded)
                gate_input = torch.cat([e_feat, rel_emb], dim=-1)
                alpha = torch.sigmoid(self.edge_mlp(gate_input))

                g.edges[rel].data["alpha"] = alpha

                # Message: gated neighbor features
                g.apply_edges(
                    fn.u_mul_e("h", "alpha", "m"),
                    etype=rel
                )

            # Aggregate messages
            g.multi_update_all(
                {
                    rel: (fn.copy_e("m", "m"), fn.sum("m", "neigh"))
                    for rel in self.rel_names if g.num_edges(rel) > 0
                },
                cross_reducer="sum"
            )

            out = {}
            for ntype in h_dict:
                h0 = h_dict[ntype]
                neigh = g.nodes[ntype].data.get(
                    "neigh", torch.zeros_like(h0)
                )

                h = (1 + self.eps[ntype]) * h0 + neigh
                h = self.node_mlp[ntype](h)

                # Residual
                out[ntype] = h + h0

            return out

class CascadedHeteroModel(BaseModel):
    def __init__(
        self,
        node_dims,
        edge_dims,
        hidden_dim,
        out_dim,
        rel_names,
        num_layers=3,
        stage="3",
        pooling_type="mean",
    ):
        super().__init__(
            node_dims=node_dims,
            edge_dims=edge_dims,
            hidden_dim=hidden_dim,
            out_dim=out_dim,
            rel_names=rel_names,
            stage=stage,
            pooling_type=pooling_type,
            model_type="EdgeDenoisedHGIN",
        )

        # Node projection layers (non-negotiable input dims)
        self.node_proj = nn.ModuleDict({
            ntype: nn.Sequential(
                nn.Linear(node_dims[ntype], hidden_dim),
                nn.LayerNorm(hidden_dim),
            )
            for ntype in node_dims
        })

        # Edge projection (shared, low-rank)
        self.edge_proj = nn.ModuleDict({
            "_".join(rel): nn.Sequential(
                nn.Linear(edge_dims[rel], 32),
                nn.LayerNorm(32),
            )
            for rel in rel_names
        })

        self.layers = nn.ModuleList([
            EdgeGatedHeteroGINLayer(
                hidden_dim=hidden_dim,
                edge_dim=32 + 16,  # edge proj + rel emb
                rel_names=rel_names,
            )
            for _ in range(num_layers)
        ])

    def forward(self, batch):
        g = batch["graph"]

        # --- Node projection ---
        h = {}
        for ntype in g.ntypes:
            x = g.nodes[ntype].data["feat"]
            h[ntype] = self.node_proj[ntype](x)

        # --- Edge projection ---
        edge_feat = {}
        for rel in self.rel_names:
            if g.num_edges(rel) == 0:
                continue
            e = g.edges[rel].data["feat"]
            edge_feat[rel] = self.edge_proj["_".join(rel)](e)

        # --- GIN layers ---
        for layer in self.layers:
            h = layer(g, h, edge_feat)

        self.h = h
        return super().forward(batch)
