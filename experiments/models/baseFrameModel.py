import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.function as fn


class BaseModel(nn.Module):
    def __init__(
        self,
        node_dims,
        edge_dims,
        hidden_dim,
        out_dim,
        rel_names,
        stage="3",
        pooling_type="mean",
        model_type="HGT"
    ):
        super().__init__()
        self.stage = str(stage)
        self.rel_names = rel_names
        self.hidden_dim = hidden_dim
        self.pooling_type = pooling_type
        self.model_type = model_type

        # Keep declared dims for runtime sanity checks and clearer errors
        self.node_dims = node_dims
        self.edge_dims = edge_dims # not used for now
        self.ntypes = list(node_dims.keys())
        self.rels = list(rel_names)

        self.h = None  # will be set by subclass

        if self.stage == "3":
            # Stage 3: Node Classification (AST/CFG Node Level)
            self.classifier = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.LayerNorm(hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim // 2, out_dim), # Multi-class 8 Logits
            )

        else:
            # Stage 1 & 2: Graph/Subgraph Classification (Contract/Function Level)

            self.classifier = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim//2),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim//2, 1),  # Binary Logit
            )

    def forward(self, batch):
        g = batch["graph"]

        # h is set by subclass

        def get_h(ntype):
            return self.h.get(ntype, torch.zeros(0, self.hidden_dim, device=g.device))

        if self.stage == "3":
            return {
                "ast_logits": self.classifier(get_h("ast_node")),
                "cfg_logits": self.classifier(get_h("cfg_node")),
            }

        else:
            ast_feats = get_h('ast_node')
            cfg_feats = get_h('cfg_node')

            # Pooling for graph-level representations
            if g.num_nodes('ast_node') > 0:
                g.nodes['ast_node'].data['h'] = ast_feats
                ast_pooled = dgl.readout_nodes(
                    g, 'h', ntype='ast_node', op='sum')
            else:
                ast_pooled = torch.zeros(
                    g.batch_size, self.hidden_dim, device=g.device)

            if g.num_nodes('cfg_node') > 0:
                g.nodes['cfg_node'].data['h'] = cfg_feats
                cfg_pooled = dgl.readout_nodes(
                    g, 'h', ntype='cfg_node', op='sum')
            else:
                cfg_pooled = torch.zeros(
                    g.batch_size, self.hidden_dim, device=g.device)

            # Shape: (batch_size, hidden_dim * 2)
            h3 = torch.cat([ast_pooled, cfg_pooled], dim=-1)

            # Shape: (batch_size, 1) - one logit per graph
            return {"graph_logits": self.classifier(h3)}
