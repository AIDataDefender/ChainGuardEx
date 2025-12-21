import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import warnings

from .baseFrameModel import BaseModel

from dgl.nn.pytorch import (
    HGTConv,
    GATv2Conv,
    GraphConv,
    SAGEConv,
    GINConv,
    GINEConv,
    GatedGCNConv,
)


SUPPORTED_MODEL_TYPES = (
    "HGT",
    "GCN",
    "GraphSAGE", #
    "GIN", #
    "GINE",
    "GatedGCN",
    "GATv2_no_edge",
    "GATv2_with_edges",
)

# ============================================================
# UNIVERSAL EDGE-AWARE CONV BASE
# ============================================================


class NodeEdgeProjector(nn.Module):
    """
    Projects src node, dst node, and edge features to hidden_dim
    GUARANTEES dimensional consistency.
    """

    def __init__(self, src_node_in, dst_node_in, edge_in, hidden_dim):
        super().__init__()
        self.src_proj = nn.Linear(src_node_in, hidden_dim)
        self.dst_proj = nn.Linear(dst_node_in, hidden_dim)
        self.edge_proj = nn.Linear(edge_in, hidden_dim)

    def forward(self, src_feat, dst_feat, edge_feat):
        """
        Project or pass-through src, dst, and edge features.

        Behavior:
        - If a feature's last dimension equals the projector's input dim, apply projection.
        - If it equals the projector's output dim (hidden dim), assume features are already
            projected and pass them through (a warning is emitted).
        - If neither, raise a ValueError with a clear message so caller can fix the declaration.
        """

        def _prepare_feat(feat, proj, name):
            if feat is None:
                raise ValueError(f"{name} feature is None")
            if feat.dim() == 1:
                # Treat scalar per-element features as (N,1)
                feat = feat.unsqueeze(-1)
            in_dim = proj.in_features
            out_dim = proj.out_features
            feat_dim = feat.size(-1)
            if feat_dim == in_dim:
                return proj(feat)
            if feat_dim == out_dim:
                warnings.warn(
                    f"{name} features already have hidden dim {out_dim}; skipping projection.")
                return feat
            raise ValueError(
                f"{name} feature dimension mismatch: got {feat_dim}, expected input dim {in_dim} "
                f"or already-projected dim {out_dim}.")

        src_h = _prepare_feat(src_feat, self.src_proj, "src")
        dst_h = _prepare_feat(dst_feat, self.dst_proj, "dst")

        # Edge features: allow scalar per-edge or pre-projected edge features
        if edge_feat is None:
            raise ValueError("edge feature is None")
        if edge_feat.dim() == 1:
            edge_feat = edge_feat.unsqueeze(-1)
        edge_in = self.edge_proj.in_features
        edge_out = self.edge_proj.out_features
        edge_dim = edge_feat.size(-1)
        if edge_dim == edge_in:
            edge_h = self.edge_proj(edge_feat)
        elif edge_dim == edge_out:
            warnings.warn(
                f"edge features already have hidden dim {edge_out}; skipping projection.")
            edge_h = edge_feat
        else:
            raise ValueError(
                f"edge feature dimension mismatch: got {edge_dim}, expected input dim {edge_in} "
                f"or already-projected dim {edge_out}.")

        return src_h, dst_h, edge_h


class NodePairProjector(nn.Module):
    """Projects src and dst node features independently to hidden_dim."""

    def __init__(self, src_node_in, dst_node_in, hidden_dim):
        super().__init__()
        self.src_proj = nn.Linear(src_node_in, hidden_dim)
        self.dst_proj = nn.Linear(dst_node_in, hidden_dim)

    def forward(self, src_feat, dst_feat):
        if src_feat is None or dst_feat is None:
            raise ValueError("src_feat and dst_feat must be provided")
        if src_feat.dim() == 1:
            src_feat = src_feat.unsqueeze(-1)
        if dst_feat.dim() == 1:
            dst_feat = dst_feat.unsqueeze(-1)
        if src_feat.size(-1) == self.src_proj.in_features:
            src_h = self.src_proj(src_feat)
        elif src_feat.size(-1) == self.src_proj.out_features:
            warnings.warn(
                f"src features already have hidden dim {self.src_proj.out_features}; skipping projection.")
            src_h = src_feat
        else:
            raise ValueError(
                f"src feature dimension mismatch: got {src_feat.size(-1)}, expected {self.src_proj.in_features} "
                f"or {self.src_proj.out_features}.")

        if dst_feat.size(-1) == self.dst_proj.in_features:
            dst_h = self.dst_proj(dst_feat)
        elif dst_feat.size(-1) == self.dst_proj.out_features:
            warnings.warn(
                f"dst features already have hidden dim {self.dst_proj.out_features}; skipping projection.")
            dst_h = dst_feat
        else:
            raise ValueError(
                f"dst feature dimension mismatch: got {dst_feat.size(-1)}, expected {self.dst_proj.in_features} "
                f"or {self.dst_proj.out_features}.")

        return src_h, dst_h


# ============================================================
# EDGE-AWARE WRAPPERS
# ============================================================

class GINEConvWrapper(nn.Module):
    def __init__(self, src_node_in, dst_node_in, edge_in, hidden_dim):
        super().__init__()
        self.proj = NodeEdgeProjector(
            src_node_in, dst_node_in, edge_in, hidden_dim)
        self.conv = GINEConv(
            nn.Identity(),  # node already projected
            hidden_dim,
        )

    def forward(self, g, feat_src, feat_dst, feat_edge):
        if feat_edge is None:
            raise ValueError("feat_edge is required for GINE")
        if feat_edge.dim() == 1:
            feat_edge = feat_edge.unsqueeze(-1)
        src_h, dst_h, edge_h = self.proj(feat_src, feat_dst, feat_edge)
        return self.conv(g, (src_h, dst_h), edge_h)


class GatedGCNConvWrapper(nn.Module):
    def __init__(self, src_node_in, dst_node_in, edge_in, hidden_dim):
        super().__init__()
        self.proj = NodeEdgeProjector(
            src_node_in, dst_node_in, edge_in, hidden_dim)
        self.conv = GatedGCNConv(
            hidden_dim,
            hidden_dim,
            edge_feats=hidden_dim,
        )

    def forward(self, g, feat_src, feat_dst, feat_edge):
        if feat_edge is None:
            raise ValueError("feat_edge is required for GatedGCN")
        if feat_edge.dim() == 1:
            feat_edge = feat_edge.unsqueeze(-1)
        src_h, dst_h, edge_h = self.proj(feat_src, feat_dst, feat_edge)
        out = self.conv(g, (src_h, dst_h), edge_h)
        if isinstance(out, tuple):
            out = out[0]
        return out


class GATv2EdgeWrapper(nn.Module):
    def __init__(self, src_node_in, dst_node_in, edge_in, hidden_dim, heads=4):
        super().__init__()
        if hidden_dim % heads != 0:
            raise ValueError(
                f"hidden_dim ({hidden_dim}) must be divisible by heads ({heads})")
        self.heads = heads
        self.proj = NodeEdgeProjector(
            src_node_in, dst_node_in, edge_in, hidden_dim)
        self.conv = GATv2Conv(
            hidden_dim,
            hidden_dim // heads,
            num_heads=heads,
            edge_feats=hidden_dim,
        )

    def forward(self, g, feat_src, feat_dst, feat_edge):
        if feat_edge is None:
            raise ValueError("feat_edge is required for edge-aware GATv2")
        if feat_edge.dim() == 1:
            feat_edge = feat_edge.unsqueeze(-1)
        src_h, dst_h, edge_h = self.proj(feat_src, feat_dst, feat_edge)
        out = self.conv(g, (src_h, dst_h), edge_h)
        # DGL GAT-style convs return (N_dst, num_heads, out_feats)
        if out.dim() == 3:
            out = out.flatten(1)
        return out


class GINConvWrapper(nn.Module):
    """GIN wrapper that is safe for bipartite relation graphs."""

    def __init__(self, src_node_in, dst_node_in, hidden_dim):
        super().__init__()
        self.proj = NodePairProjector(src_node_in, dst_node_in, hidden_dim)
        # Keep a learnable transformation after aggregation.
        self.conv = GINConv(nn.Linear(hidden_dim, hidden_dim))

    def forward(self, g, feat_src, feat_dst):
        src_h, dst_h = self.proj(feat_src, feat_dst)
        out = self.conv(g, (src_h, dst_h))
        return out


# ============================================================
# MAIN MODEL
# ============================================================

class CascadedHeteroModel(BaseModel):
    def __init__(
        self,
        node_dims,
        edge_dims,
        hidden_dim,
        out_dim,
        rel_names,
        stage="3",
        model_type="HGT",
    ):
        super().__init__(node_dims, edge_dims, hidden_dim, out_dim, rel_names, stage, "mean", model_type)
        # ----------------------------------------------------
        # Backbone
        # ----------------------------------------------------
        print("="*40,"\n\n")
        print(f"Using {model_type.upper()} - Stage {self.stage}")
        print("="*40,"\n\n")
        if model_type == "HGT": # OK
            # DGL's HGTConv operates on a homogeneous graph and requires explicit
            # node-type and edge-type ID tensors in forward().
            # To support heterogeneous input node feature dimensions, we project
            # each ntype to a shared hidden_dim first.
            self.hgt_in = nn.ModuleDict(
                {nt: nn.Linear(node_dims[nt], hidden_dim)
                    for nt in self.ntypes}
            )
            self.gnn = HGTConv(
                in_size=hidden_dim,
                head_size=hidden_dim // 4,
                num_heads=4,
                num_ntypes=len(self.ntypes),
                num_etypes=len(self.rels),
                dropout=0.1,
                use_norm=True,
            )
        else:
            self.layer1 = nn.ModuleDict()
            self.layer2 = nn.ModuleDict()
            self.layer3 = nn.ModuleDict()

            for rel in self.rels:
                srctype, etype, dsttype = rel
                src_in = node_dims[srctype]
                dst_in = node_dims[dsttype]
                e_in = edge_dims[rel]
                rel_key = "_".join(rel)

                if model_type == "GCN": # OK
                    self.layer1[rel_key] = GraphConv(
                        (src_in, dst_in), hidden_dim)
                    self.layer2[rel_key] = GraphConv(
                        (hidden_dim, hidden_dim), hidden_dim)
                    self.layer3[rel_key] = GraphConv(
                        (hidden_dim, hidden_dim), hidden_dim)
                elif model_type == "GraphSAGE": # OK
                    self.layer1[rel_key] = SAGEConv(
                        (src_in, dst_in), hidden_dim, aggregator_type='mean')
                    self.layer2[rel_key] = SAGEConv(
                        (hidden_dim, hidden_dim), hidden_dim, aggregator_type='mean')
                    self.layer3[rel_key] = SAGEConv(
                        (hidden_dim, hidden_dim), hidden_dim, aggregator_type='mean')
                elif model_type == "GIN": # OK
                    self.layer1[rel_key] = GINConvWrapper(
                        src_in, dst_in, hidden_dim)
                    self.layer2[rel_key] = GINConvWrapper(
                        hidden_dim, hidden_dim, hidden_dim)
                    self.layer3[rel_key] = GINConvWrapper(
                        hidden_dim, hidden_dim, hidden_dim)
                elif model_type == "GINE": #OK
                    self.layer1[rel_key] = GINEConvWrapper(
                        src_in, dst_in, e_in, hidden_dim)
                    self.layer2[rel_key] = GINEConvWrapper(
                        hidden_dim, hidden_dim, e_in, hidden_dim)
                    self.layer3[rel_key] = GINEConvWrapper(
                        hidden_dim, hidden_dim, e_in, hidden_dim)
                elif model_type == "GatedGCN": # OK
                    self.layer1[rel_key] = GatedGCNConvWrapper(
                        src_in, dst_in, e_in, hidden_dim)
                    self.layer2[rel_key] = GatedGCNConvWrapper(
                        hidden_dim, hidden_dim, e_in, hidden_dim)
                    self.layer3[rel_key] = GatedGCNConvWrapper(
                        hidden_dim, hidden_dim, e_in, hidden_dim)
                elif model_type == "GATv2_no_edge": # OK
                    heads = 4
                    if hidden_dim % heads != 0:
                        raise ValueError(
                            f"hidden_dim ({hidden_dim}) must be divisible by heads ({heads})")
                    self.layer1[rel_key] = GATv2Conv(
                        (src_in, dst_in), hidden_dim // heads, num_heads=heads, allow_zero_in_degree=True)
                    self.layer2[rel_key] = GATv2Conv(
                        (hidden_dim, hidden_dim), hidden_dim // heads, num_heads=heads, allow_zero_in_degree=True)
                    self.layer3[rel_key] = GATv2Conv(
                        (hidden_dim, hidden_dim), hidden_dim // heads, num_heads=heads, allow_zero_in_degree=True)
                elif model_type == "GATv2_with_edges": # OK
                    self.layer1[rel_key] = GATv2EdgeWrapper(
                        src_in, dst_in, e_in, hidden_dim, allow_zero_in_degree=True)
                    self.layer2[rel_key] = GATv2EdgeWrapper(
                        hidden_dim, hidden_dim, e_in, hidden_dim, allow_zero_in_degree=True)
                    self.layer3[rel_key] = GATv2EdgeWrapper(
                        hidden_dim, hidden_dim, e_in, hidden_dim, allow_zero_in_degree=True)
                else:
                    raise ValueError(f"Unknown model_type: {model_type}")

        # ----------------------------------------------------
        # Heads
        # ----------------------------------------------------

        # Handled by BaseModel

    # --------------------------------------------------------
    # Forward
    # --------------------------------------------------------
    
    def forward(self, batch):
        g = batch["graph"]

        if self.model_type == "HGT":
            # 1) Per-ntype input projection to a shared dimension
            for nt in self.ntypes:
                if nt not in g.ntypes:
                    raise ValueError(f"Graph is missing expected ntype: {nt}")
                g.nodes[nt].data["_hgt_in"] = self.hgt_in[nt](
                    g.nodes[nt].data["feat"])

            # 2) Convert to homogeneous and remap type IDs to our configured order
            hg = dgl.to_homogeneous(g, ndata=["_hgt_in"])
            x = hg.ndata["_hgt_in"]

            # Raw IDs from DGL correspond to g.ntypes and g.canonical_etypes ordering.
            raw_ntype = hg.ndata[dgl.NTYPE]
            raw_etype = hg.edata[dgl.ETYPE]

            ntype_map = torch.empty(
                len(g.ntypes), dtype=torch.long, device=x.device)
            for i, ntype_name in enumerate(g.ntypes):
                try:
                    ntype_map[i] = self.ntypes.index(ntype_name)
                except ValueError as e:
                    raise ValueError(
                        f"Encountered graph ntype {ntype_name} not present in model ntypes {self.ntypes}"
                    ) from e
            ntype = ntype_map[raw_ntype]

            etype_map = torch.empty(
                len(g.canonical_etypes), dtype=torch.long, device=x.device)
            for i, etype_name in enumerate(g.canonical_etypes):
                try:
                    etype_map[i] = self.rels.index(etype_name)
                except ValueError as e:
                    raise ValueError(
                        f"Encountered graph canonical etype {etype_name} not present in model rels {self.rels}"
                    ) from e
            etype = etype_map[raw_etype]

            # 3) HGTConv forward on homogeneous graph
            out = self.gnn(hg, x, ntype, etype)

            # 4) Scatter back to per-ntype tensors
            h = {}
            raw_nid = hg.ndata[dgl.NID]
            for model_tid, ntype_name in enumerate(self.ntypes):
                out_nt = torch.zeros(
                    g.num_nodes(ntype_name), self.hidden_dim, device=out.device, dtype=out.dtype
                )
                mask = ntype == model_tid
                if mask.any():
                    out_nt[raw_nid[mask]] = out[mask]
                h[ntype_name] = out_nt
        else:
            h = {nt: g.nodes[nt].data["feat"] for nt in self.ntypes}
            # Layer 1
            h1 = {ntype: torch.zeros(g.num_nodes(
                ntype), self.hidden_dim, device=g.device) for ntype in g.ntypes}
            for rel in self.rels:
                srctype, etype, dsttype = rel
                rel_key = "_".join(rel)
                if rel not in g.canonical_etypes or g.num_edges(rel) == 0:
                    continue
                src_feat = g.nodes[srctype].data["feat"]
                dst_feat = g.nodes[dsttype].data["feat"]
                edge_feat = g.edges[rel].data["feat"]
                g_rel = g[rel]
                if hasattr(self.layer1[rel_key], 'proj'):
                    # Edge-aware wrappers take explicit src/dst/edge
                    if isinstance(self.layer1[rel_key], GINConvWrapper):
                        res = self.layer1[rel_key](g_rel, src_feat, dst_feat)
                    else:
                        res = self.layer1[rel_key](
                            g_rel, src_feat, dst_feat, edge_feat)
                else:
                    # DGL convs on relation graphs must receive (feat_src, feat_dst)
                    res = self.layer1[rel_key](g_rel, (src_feat, dst_feat))
                if isinstance(res, tuple):
                    res = res[0]
                if res.dim() == 3:
                    res = res.flatten(1)
                h1[dsttype] += res
            h1 = {k: F.relu(v) for k, v in h1.items()}

            # Layer 2
            h2 = {ntype: torch.zeros_like(h1[ntype]) for ntype in g.ntypes}
            for rel in self.rels:
                srctype, etype, dsttype = rel
                rel_key = "_".join(rel)
                if rel not in g.canonical_etypes or g.num_edges(rel) == 0:
                    continue
                edge_feat = g.edges[rel].data["feat"]
                g_rel = g[rel]
                if hasattr(self.layer2[rel_key], 'proj'):
                    if isinstance(self.layer2[rel_key], GINConvWrapper):
                        res = self.layer2[rel_key](
                            g_rel, h1[srctype], h1[dsttype])
                    else:
                        res = self.layer2[rel_key](
                            g_rel, h1[srctype], h1[dsttype], edge_feat)
                else:
                    res = self.layer2[rel_key](
                        g_rel, (h1[srctype], h1[dsttype]))
                if isinstance(res, tuple):
                    res = res[0]
                if res.dim() == 3:
                    res = res.flatten(1)
                h2[dsttype] += res
            h2 = {k: F.relu(v) for k, v in h2.items()}

            # Layer 3
            h3 = {ntype: torch.zeros_like(h2[ntype]) for ntype in g.ntypes}
            for rel in self.rels:
                srctype, etype, dsttype = rel
                rel_key = "_".join(rel)
                if rel not in g.canonical_etypes or g.num_edges(rel) == 0:
                    continue
                edge_feat = g.edges[rel].data["feat"]
                g_rel = g[rel]
                if hasattr(self.layer3[rel_key], 'proj'):
                    if isinstance(self.layer3[rel_key], GINConvWrapper):
                        res = self.layer3[rel_key](
                            g_rel, h2[srctype], h2[dsttype])
                    else:
                        res = self.layer3[rel_key](
                            g_rel, h2[srctype], h2[dsttype], edge_feat)
                else:
                    res = self.layer3[rel_key](
                        g_rel, (h2[srctype], h2[dsttype]))
                if isinstance(res, tuple):
                    res = res[0]
                if res.dim() == 3:
                    res = res.flatten(1)
                h3[dsttype] += res
            h3 = {k: F.relu(v) for k, v in h3.items()}
            h = h3
            # Residual from Layer 1 to Layer 3
            # h = {k: v + h1[k] for k, v in h3.items()}

        self.h = h
        return super().forward(batch)
