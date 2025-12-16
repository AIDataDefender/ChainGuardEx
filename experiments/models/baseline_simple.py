import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl.nn import GraphConv, HeteroGraphConv, GATConv


class SimpleHeteroModel(nn.Module):
    """
    Simpler, lighter variant of CascadedHeteroModel.
    - Same inputs/outputs as baseline_5.CascadedHeteroModel
    - HeteroGraphConv + mean pooling (no multi-head attention)
    - Small MLP heads
    """

    def __init__(self, node_dims, edge_dims, hidden_dim, out_dim, rel_names, stage="3"):
        super().__init__()
        self.rel_names = rel_names
        self.stage = str(stage)
        self.hidden_dim = hidden_dim

        # Two-layer hetero GNN with lightweight attention (2 heads) per relation
        self.heads = 2
        def _in_feats_tuple(rel):
            src_dim = node_dims.get(rel[0], 128)
            dst_dim = node_dims.get(rel[2], src_dim)
            return (src_dim, dst_dim)

        self.layer1 = HeteroGraphConv(
            {
                rel: GATConv(
                    in_feats=_in_feats_tuple(rel),
                    out_feats=hidden_dim // self.heads,
                    num_heads=self.heads,
                    feat_drop=0.1,
                    attn_drop=0.1,
                    allow_zero_in_degree=True,
                )
                for rel in rel_names
            },
            aggregate="mean",
        )
        self.layer2 = HeteroGraphConv(
            {
                rel: GATConv(
                    in_feats=(hidden_dim, hidden_dim),
                    out_feats=hidden_dim // self.heads,
                    num_heads=self.heads,
                    feat_drop=0.1,
                    attn_drop=0.1,
                    allow_zero_in_degree=True,
                )
                for rel in rel_names
            },
            aggregate="mean",
        )
        self.dropout = nn.Dropout(0.2)

        if self.stage == "3":
            # Compact heads for AST/CFG nodes
            self.classify_ast = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim // 2, out_dim),
            )
            self.classify_cfg = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim // 2, out_dim),
            )
        else:
            # Attention pooling per type, then concat
            self.ast_gate = nn.Linear(hidden_dim, 1)
            self.cfg_gate = nn.Linear(hidden_dim, 1)
            self.classifier = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim // 2, 1),
            )

    def _get_feats(self, g):
        feats = {}
        for ntype in g.ntypes:
            if "feat" in g.nodes[ntype].data:
                feats[ntype] = g.nodes[ntype].data["feat"]
            else:
                # Fallback to zeros if missing
                dim = g.nodes[ntype].data.get("feat", torch.zeros(
                    0)).shape[-1] if g.nodes[ntype].data else 0
                feats[ntype] = torch.zeros(
                    g.num_nodes(ntype), dim, device=g.device)
        return feats

    def forward(self, batch_dict):
        g = batch_dict["graph"]
        h = self._get_feats(g)

        def _combine(h_dict):
            out = {}
            for k, v in h_dict.items():
                # v: [N, heads, out_dim]; flatten heads like baseline_5
                if v.dim() == 3:
                    v = v.flatten(1)  # [N, heads*out_dim]
                out[k] = self.dropout(F.relu(v))
            return out

        h1 = _combine(self.layer1(g, h))
        h2 = _combine(self.layer2(g, h1))

        def get_h(ntype):
            return h2.get(ntype, torch.zeros(0, self.hidden_dim, device=g.device))

        if self.stage == "3":
            return {
                "ast_logits": self.classify_ast(get_h("ast_node")),
                "cfg_logits": self.classify_cfg(get_h("cfg_node")),
            }
        else:
            # Attention pool per type to batch
            def attn_pool(ntype, gate):
                if g.num_nodes(ntype) == 0:
                    return torch.zeros(g.batch_size, self.hidden_dim, device=g.device)
                h_nt = get_h(ntype)
                score = gate(h_nt)
                g.nodes[ntype].data["h"] = h_nt
                g.nodes[ntype].data["a"] = score
                g.nodes[ntype].data["a"] = dgl.softmax_nodes(
                    g, "a", ntype=ntype)
                g.nodes[ntype].data["h_weighted"] = h_nt * \
                    g.nodes[ntype].data["a"]
                return dgl.readout_nodes(g, "h_weighted", ntype=ntype, op="sum")

            ast_pooled = attn_pool("ast_node", self.ast_gate)
            cfg_pooled = attn_pool("cfg_node", self.cfg_gate)
            h_cat = torch.cat([ast_pooled, cfg_pooled], dim=-1)
            return {"graph_logits": self.classifier(h_cat)}
