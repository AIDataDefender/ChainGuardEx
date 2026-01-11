import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn
from .baseFrameModel import BaseModel


class HGINAConv(nn.Module):
    """
    HGINA: Heterogeneous GIN with minimal relation-level attention

    Key properties:
    - GIN-style sum aggregation (injective, robust)
    - Scalar attention per relation (not per edge, not per head)
    - No edge feature dependence (optional extension later)
    """

    def __init__(
        self,
        in_node_feats_src,
        in_node_feats_dst,
        out_feats,
        dropout=0.1,
        residual=True,
        eps_init=0.0,
        mlp_hidden_multiplier=2,
    ):
        super().__init__()

        self.out_feats = out_feats
        self.residual = residual

        # ---- Linear projection (align src → dst space) ----
        self.lin_src = nn.Linear(in_node_feats_src, out_feats, bias=False)
        self.lin_dst = nn.Linear(in_node_feats_dst, out_feats, bias=False)

        # ---- GIN epsilon ----
        self.eps = nn.Parameter(torch.tensor(eps_init))

        # ---- Relation-level attention (scalar) ----
        # One learnable weight per relation (this module = one relation)
        self.rel_attn = nn.Parameter(torch.tensor(1.0))

        # ---- GIN MLP ----
        hidden_dim = mlp_hidden_multiplier * out_feats
        self.mlp = nn.Sequential(
            nn.Linear(out_feats, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_feats),
        )

        self.norm = nn.LayerNorm(out_feats)
        self.dropout = nn.Dropout(dropout)

        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
        nn.init.constant_(self.rel_attn, 1.0)

    def forward(self, graph, feat_src, feat_dst):
        """
        graph    : DGLGraph (single canonical relation)
        feat_src : [N_src, in_node_feats_src]
        feat_dst : [N_dst, in_node_feats_dst]
        """

        with graph.local_scope():

            h_src = self.lin_src(feat_src)
            h_dst = self.lin_dst(feat_dst)

            graph.srcdata["h"] = h_src

            # ---- Pure GIN aggregation (sum) ----
            graph.update_all(
                fn.copy_u("h", "m"),
                fn.sum("m", "agg")
            )

            agg = graph.dstdata["agg"]

            # ---- Minimal relation attention (scalar) ----
            agg = self.rel_attn * agg

            # ---- GIN update ----
            out = (1 + self.eps) * h_dst + agg
            out = self.mlp(self.norm(out))
            out = self.dropout(out)

            if self.residual and feat_dst.shape[1] == self.out_feats:
                out = out + feat_dst

            return out


class HGINAStack(nn.Module):
    """
    HGINA: Heterogeneous GIN with minimal attention (relation-wise)
    """

    def __init__(
        self,
        rel_names,
        node_dims,
        hidden_dim,
        num_layers=3,
        dropout=0.1,
        use_jk=True,
    ):
        super().__init__()

        self.rel_names = rel_names
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.use_jk = use_jk

        self.layers = nn.ModuleList()

        for layer_idx in range(num_layers):
            layer = nn.ModuleDict()
            for rel in rel_names:
                srctype, etype, dsttype = rel
                rel_key = "_".join(rel)

                in_src = node_dims[srctype] if layer_idx == 0 else hidden_dim
                in_dst = node_dims[dsttype] if layer_idx == 0 else hidden_dim

                layer[rel_key] = HGINAConv(
                    in_node_feats_src=in_src,
                    in_node_feats_dst=in_dst,
                    out_feats=hidden_dim,
                    dropout=dropout,
                )

            self.layers.append(layer)

    def forward(self, g):
        """
        g : DGLHeteroGraph
        Assumes g.nodes[ntype].data["feat"] exists
        """

        h = {ntype: g.nodes[ntype].data["feat"] for ntype in g.ntypes}
        h_list = []

        for layer in self.layers:
            h_new = {
                ntype: torch.zeros(
                    h[ntype].shape[0],
                    self.hidden_dim,
                    device=h[ntype].device,
                    dtype=h[ntype].dtype,
                )
                for ntype in g.ntypes
            }

            for rel in self.rel_names:
                srctype, etype, dsttype = rel
                rel_key = "_".join(rel)

                if rel not in g.canonical_etypes or g.num_edges(rel) == 0:
                    continue

                out = layer[rel_key](
                    g[rel],
                    h[srctype],
                    h[dsttype],
                )

                h_new[dsttype] += out

            h = {k: F.relu(v) for k, v in h_new.items()}
            h_list.append(h)

        if self.use_jk:
            h = {
                k: sum(layer_h[k] for layer_h in h_list)
                for k in h
            }

        return h


class HeteroRGCNLayer(nn.Module):
    """
    Relation-aware Graph Convolution for Heterogeneous Graphs
    Used in Stage 1 & 2 (scalable + stable)
    """

    def __init__(self, in_dim, out_dim, rel_names, ntypes=None, dropout=0.1, eps_init=0.0):
        super().__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.rel_names = rel_names
        self.ntypes = list(ntypes) if ntypes is not None else None

        # Relation-specific projection: src -> out_dim
        self.weight = nn.ModuleDict(
            {"_".join(rel): nn.Linear(in_dim, out_dim, bias=False)
            for rel in rel_names}
        )

        # GIN-style epsilon on the self term
        self.eps = nn.Parameter(torch.tensor(float(eps_init)))

        # Per-node-type self loop and residual projection
        if self.ntypes is None:
            self.self_loop = nn.Linear(in_dim, out_dim, bias=True)
            self.res_proj = (
                nn.Identity()
                if in_dim == out_dim
                else nn.Linear(in_dim, out_dim, bias=False)
            )
        else:
            self.self_loop = nn.ModuleDict(
                {nt: nn.Linear(in_dim, out_dim, bias=True)
                for nt in self.ntypes}
            )
            self.res_proj = nn.ModuleDict(
                {
                    nt: (nn.Identity() if in_dim == out_dim else nn.Linear(
                        in_dim, out_dim, bias=False))
                    for nt in self.ntypes
                }
            )

        # Per-dst-type relation mixing (softmax over incoming relations)
        # Logit = Linear(tanh([h_self || neigh_rel]))
        if self.ntypes is None:
            self.rel_attn = nn.Linear(2 * out_dim, 1, bias=False)
        else:
            self.rel_attn = nn.ModuleDict(
                {nt: nn.Linear(2 * out_dim, 1, bias=False)
                    for nt in self.ntypes}
            )

        hidden_dim = 2 * out_dim
        if self.ntypes is None:
            self.norm = nn.LayerNorm(out_dim)
            self.ffn = nn.Sequential(
                nn.Linear(out_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, out_dim),
            )
        else:
            self.norm = nn.ModuleDict(
                {nt: nn.LayerNorm(out_dim) for nt in self.ntypes})
            self.ffn = nn.ModuleDict(
                {
                    nt: nn.Sequential(
                        nn.Linear(out_dim, hidden_dim),
                        nn.GELU(),
                        nn.Dropout(dropout),
                        nn.Linear(hidden_dim, out_dim),
                    )
                    for nt in self.ntypes
                }
            )

        self.feat_drop = nn.Dropout(dropout)
        self.dropout = nn.Dropout(dropout)

        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, g, h_dict):
        with g.local_scope():
            # 1) Compute per-relation neighborhood messages (sum aggregation)
            neigh_by_rel = {}
            for rel in self.rel_names:
                srctype, _, _ = rel
                rel_key = "_".join(rel)

                if rel not in g.canonical_etypes or g.num_edges(rel) == 0:
                    continue

                Wh = self.weight[rel_key](self.feat_drop(h_dict[srctype]))
                rel_g = g[rel]
                rel_g.srcdata["Wh"] = Wh
                rel_g.update_all(fn.copy_u("Wh", "m"), fn.sum("m", "neigh"))
                neigh_by_rel[rel_key] = rel_g.dstdata["neigh"]

            # 2) Mix relations per destination node type with a softmax attention
            h_out = {}
            for ntype in g.ntypes:
                num_nodes = g.num_nodes(ntype)
                if num_nodes == 0:
                    h_out[ntype] = torch.zeros(
                        0,
                        self.out_dim,
                        device=g.device,
                    )
                    continue

                if self.ntypes is None:
                    h_self = self.self_loop(self.feat_drop(h_dict[ntype]))
                    res = self.res_proj(h_dict[ntype])
                    norm = self.norm
                    ffn = self.ffn
                    rel_attn = self.rel_attn
                else:
                    h_self = self.self_loop[ntype](
                        self.feat_drop(h_dict[ntype]))
                    res = self.res_proj[ntype](h_dict[ntype])
                    norm = self.norm[ntype]
                    ffn = self.ffn[ntype]
                    rel_attn = self.rel_attn[ntype]

                # Collect incoming relation messages for this destination type
                in_rels = [rel for rel in self.rel_names if rel[2] == ntype]
                if len(in_rels) == 0:
                    neigh_mix = torch.zeros_like(h_self)
                else:
                    msg_list = []
                    present_mask = []
                    for rel in in_rels:
                        rel_key = "_".join(rel)
                        if rel_key in neigh_by_rel:
                            msg_list.append(neigh_by_rel[rel_key])
                            present_mask.append(True)
                        else:
                            msg_list.append(
                                torch.zeros(
                                    num_nodes,
                                    self.out_dim,
                                    device=h_self.device,
                                    dtype=h_self.dtype,
                                )
                            )
                            present_mask.append(False)

                    msg_stack = torch.stack(msg_list, dim=1)  # [N, R, D]
                    h_rep = h_self.unsqueeze(
                        1).expand(-1, msg_stack.shape[1], -1)
                    attn_in = torch.cat(
                        [h_rep, msg_stack], dim=-1)  # [N, R, 2D]
                    logits = rel_attn(torch.tanh(attn_in)).squeeze(-1)  # [N, R]

                    if not all(present_mask):
                        mask_t = torch.tensor(
                            present_mask,
                            device=logits.device,
                            dtype=torch.bool,
                        ).unsqueeze(0)
                        logits = logits.masked_fill(~mask_t, float("-inf"))

                    alpha = F.softmax(logits, dim=1)
                    alpha = torch.nan_to_num(alpha, nan=0.0)
                    neigh_mix = (alpha.unsqueeze(-1) * msg_stack).sum(dim=1)

                # 3) GIN-style update + FFN + residual
                x = (1 + self.eps) * h_self + neigh_mix
                x = norm(x)
                x = x + ffn(x)
                x = x + res
                h_out[ntype] = self.dropout(F.gelu(x))

            return h_out


class HeteroRGCN(nn.Module):
    """
    Used in Stage 1 & 2 with GraphSAINT sampling
    """

    def __init__(
        self,
        in_dim,
        hidden_dim,
        num_layers,
        rel_names,
        ntypes=None,
        dropout=0.1,
    ):
        super().__init__()

        self.layers = nn.ModuleList()
        self.layers.append(
            HeteroRGCNLayer(in_dim, hidden_dim, rel_names,
                            ntypes=ntypes, dropout=dropout)
        )

        for _ in range(num_layers - 1):
            self.layers.append(
                HeteroRGCNLayer(hidden_dim, hidden_dim, rel_names,
                                ntypes=ntypes, dropout=dropout)
            )

    def forward(self, g, h_dict):
        for layer in self.layers:
            h_dict = layer(g, h_dict)
        return h_dict


class CascadedHeteroModel(BaseModel):
    def __init__(
        self,
        node_dims,
        edge_dims,
        hidden_dim,
        out_dim,
        rel_names,
        stage="3",
    ):
        super().__init__(node_dims, edge_dims, hidden_dim, out_dim, rel_names, stage)
        self.model_type = "HGINA_RGCN_Cascade"
        print("="*40, "\n\n")
        print(f"Using HGINA + RGCN - Stage {self.stage}")
        print("="*40, "\n\n")

        if self.stage in ["1", "2"]:
            self.hgina_stack = HGINAStack(
                rel_names=rel_names,
                node_dims=node_dims,
                hidden_dim=hidden_dim,
                num_layers=3,
            )
        else:
            self.node_proj = nn.ModuleDict({
                ntype: nn.Linear(node_dims[ntype], hidden_dim)
                for ntype in node_dims
            })
            self.rgcn = HeteroRGCN(
                in_dim=hidden_dim,
                hidden_dim=hidden_dim,
                num_layers=3,  # adjust as needed
                rel_names=rel_names,
                ntypes=list(node_dims.keys()),
            )

    def forward(self, batch_dict):
        g = batch_dict["graph"]

        if self.stage in ["1", "2"]:
            self.h = self.hgina_stack(g)
        else:
            h_dict = {
                ntype: self.node_proj[ntype](g.nodes[ntype].data["feat"])
                for ntype in g.ntypes
            }
            self.h = self.rgcn(g, h_dict)

        return super().forward(batch_dict)
