import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.function as fn
from dgl.ops import edge_softmax
from .baseFrameModel import BaseModel


class HGTEConv(nn.Module):
    """
    HGTE: Heterogeneous Graph Transformer with Edge-gated Messages

    Core:
    - Dot-product attention
    - Edge-gated messages
    - NormSoftmax aggregation
    - Pre-norm + residual
    - Feed-forward expansion

    """

    def __init__(
        self,
        in_node_feats_src,
        in_node_feats_dst,
        in_edge_feats,
        out_feats,
        num_heads=2,
        dropout=0.1,
        residual=True,
        ffn_hidden_multiplier=2,   # OPTIONAL
    ):
        super().__init__()

        assert out_feats % num_heads == 0
        self.out_feats = out_feats
        self.num_heads = num_heads
        self.head_dim = out_feats // num_heads
        self.residual = residual
        # Head-wise relation scaling (one scalar per head)
        self.rel_head_scale = nn.Parameter(torch.ones(num_heads))

        # -------- Node projections --------
        self.W_q = nn.Linear(in_node_feats_dst, out_feats, bias=False)
        self.W_k = nn.Linear(in_node_feats_src, out_feats, bias=False)
        self.W_h = nn.Linear(in_node_feats_src, out_feats, bias=False)

        # -------- Edge gate --------
        self.W_g = nn.Linear(in_edge_feats, out_feats, bias=True)

        # -------- Message MLP --------
        self.msg_mlp = nn.Sequential(
            nn.Linear(out_feats, out_feats),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(out_feats, out_feats),
        )

        # -------- Update MLP --------
        self.update_mlp = nn.Sequential(
            nn.Linear(out_feats, out_feats),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(out_feats, out_feats),
        )

        # -------- OPTIONAL Feed-Forward Expansion (Transformer FFN) --------
        self.ffn = nn.Sequential(
            nn.Linear(out_feats, ffn_hidden_multiplier * out_feats),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_hidden_multiplier * out_feats, out_feats),
        )
        self.ffn_norm = nn.LayerNorm(out_feats)

        # -------- Normalization --------
        self.norm = nn.LayerNorm(out_feats)

        self.feat_drop = nn.Dropout(dropout)
        self.attn_drop = nn.Dropout(dropout)

        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, graph, feat_src, feat_dst, feat_edge):
        """
        graph     : DGLGraph
        feat_src  : [num_src_nodes, in_node_feats_src]
        feat_dst  : [num_dst_nodes, in_node_feats_dst]
        feat_edge : [num_edges, in_edge_feats]
        """

        with graph.local_scope():

            # -------- Linear projections --------
            Q = self.W_q(self.feat_drop(feat_dst))
            K = self.W_k(self.feat_drop(feat_src))
            H = self.W_h(self.feat_drop(feat_src))

            Q = Q.view(-1, self.num_heads, self.head_dim)
            K = K.view(-1, self.num_heads, self.head_dim)
            H = H.view(-1, self.num_heads, self.head_dim)

            # -------- Edge gate --------
            gate = torch.sigmoid(self.W_g(feat_edge))
            gate = gate.view(-1, self.num_heads, self.head_dim)

            graph.dstdata["Q"] = Q
            graph.srcdata["K"] = K
            graph.srcdata["H"] = H
            graph.edata["gate"] = gate

            # -------- Dot-product attention --------
            def edge_attention(edges):
                score = (edges.dst["Q"] * edges.src["K"]).sum(dim=-1)
                score = score / (self.head_dim ** 0.5)
                return {"score": score}

            graph.apply_edges(edge_attention)

            graph.edata["alpha"] = edge_softmax(graph, graph.edata["score"])
            graph.edata["alpha"] = self.attn_drop(graph.edata["alpha"])

            # -------- Message passing --------
            def message_func(edges):
                msg = edges.src["H"] * edges.data["gate"]
                msg = msg.view(msg.shape[0], -1)  # [num_edges, out_feats]
                msg = self.msg_mlp(msg)
                # [num_edges, num_heads, head_dim]
                msg = msg.view(msg.shape[0], self.num_heads, self.head_dim)
                # Head-wise relation scaling
                msg = msg * self.rel_head_scale.view(1, self.num_heads, 1)

                msg = msg * edges.data["alpha"].unsqueeze(-1)

                return {"m": msg}

            graph.update_all(message_func, fn.sum("m", "agg"))

            # -------- Aggregate & update --------
            agg = graph.dstdata["agg"].reshape(-1, self.out_feats)

            out = self.update_mlp(self.norm(agg))

            if self.residual and feat_dst.shape[1] == self.out_feats:
                out = out + feat_dst

            # -------- OPTIONAL Feed-Forward Expansion --------
            out = out + self.ffn(self.ffn_norm(out))

            return out


class HGTEStack(nn.Module):
    """
    Stack of HGTEConv layers for Stage 3
    """

    def __init__(
        self,
        rel_names,
        node_dims,
        edge_dims,
        hidden_dim,
        num_layers=3,
        num_heads=4,
        dropout=0.1,
    ):
        super().__init__()

        self.rel_names = rel_names
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.layers = nn.ModuleList()

        for layer_idx in range(num_layers):
            layer = nn.ModuleDict()
            for rel in rel_names:
                srctype, etype, dsttype = rel
                rel_key = "_".join(rel)

                in_src = node_dims[srctype] if layer_idx == 0 else hidden_dim
                in_dst = node_dims[dsttype] if layer_idx == 0 else hidden_dim

                layer[rel_key] = HGTEConv(
                    in_node_feats_src=in_src,
                    in_node_feats_dst=in_dst,
                    in_edge_feats=edge_dims.get(rel, 64),
                    out_feats=hidden_dim,
                    num_heads=num_heads,
                    dropout=dropout,
                )
            self.layers.append(layer)

    def forward(self, g):
        """
        g : DGLHeteroGraph
        Stage 3
        """

        h = {ntype: g.nodes[ntype].data["feat"] for ntype in g.ntypes}

        h_list = []  # for optional Jump Knowledge

        for layer in self.layers:
            h_new = {ntype: torch.zeros(h[ntype].shape[0], self.hidden_dim,
                                        device=h[ntype].device, dtype=h[ntype].dtype) for ntype in g.ntypes}

            for rel in self.rel_names:
                srctype, etype, dsttype = rel
                rel_key = "_".join(rel)

                if rel not in g.canonical_etypes or g.num_edges(rel) == 0:
                    continue

                edge_feat = g.edges[rel].data["feat"]

                out = layer[rel_key](
                    g[rel],
                    h[srctype],
                    h[dsttype],
                    edge_feat,
                )

                h_new[dsttype] += out

            h = {k: F.gelu(v) for k, v in h_new.items()}
            h_list.append(h)

        # Jump Knowledge
        h = {k: sum(layer_h[k] for layer_h in h_list) for k in h}

        return h


class HeteroRGCNLayer(nn.Module):
    """
    Relation-aware Graph Convolution for Heterogeneous Graphs
    Used in Stage 1 & 2 (scalable + stable)
    """

    def __init__(self, in_dim, out_dim, rel_names, dropout=0.1):
        super().__init__()

        self.rel_names = rel_names
        self.weight = nn.ModuleDict({
            "_".join(rel): nn.Linear(in_dim, out_dim, bias=False)
            for rel in rel_names
        })

        self.self_loop = nn.Linear(in_dim, out_dim, bias=True)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(out_dim)

        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, g, h_dict):
        with g.local_scope():

            for ntype in g.ntypes:
                g.nodes[ntype].data["h"] = h_dict[ntype]

            for rel in self.rel_names:
                srctype, etype, dsttype = rel
                rel_key = "_".join(rel)

                if rel not in g.canonical_etypes or g.num_edges(rel) == 0:
                    continue

                Wh = self.weight[rel_key](g.nodes[srctype].data["h"])
                g.nodes[srctype].data[f"Wh_{rel_key}"] = Wh

                g.apply_edges(
                    fn.copy_u(f"Wh_{rel_key}", "m"),
                    etype=rel
                )
                g.update_all(
                    fn.copy_e("m", "m"),
                    fn.mean("m", "neigh"),
                    etype=rel
                )

            h_out = {}
            for ntype in g.ntypes:
                neigh = torch.zeros(
                    g.num_nodes(ntype),
                    self.self_loop.out_features,
                    device=g.device,
                )

                for rel in self.rel_names:
                    _, _, dsttype = rel
                    if dsttype == ntype and "neigh" in g.nodes[ntype].data:
                        neigh += g.nodes[ntype].data["neigh"]

                h_out[ntype] = self.norm(
                    self.self_loop(h_dict[ntype]) + neigh
                )

            return {k: self.dropout(F.relu(v)) for k, v in h_out.items()}


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
        dropout=0.1,
    ):
        super().__init__()

        self.layers = nn.ModuleList()
        self.layers.append(
            HeteroRGCNLayer(in_dim, hidden_dim, rel_names, dropout)
        )

        for _ in range(num_layers - 1):
            self.layers.append(
                HeteroRGCNLayer(hidden_dim, hidden_dim, rel_names, dropout)
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
        print("="*40,"\n\n")
        print(f"Using HGTE + RGCN - Stage {self.stage}")
        print("="*40,"\n\n")
        if self.stage == "3":
            self.hgte_stack = HGTEStack(
                rel_names=rel_names,
                node_dims=node_dims,
                edge_dims=edge_dims,
                hidden_dim=hidden_dim,
                num_layers=3,
                num_heads=4,
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
            )

    def forward(self, batch_dict):
        g = batch_dict["graph"]

        if self.stage == "3":
            self.h = self.hgte_stack(g)
        else:
            h_dict = {
                ntype: self.node_proj[ntype](g.nodes[ntype].data["feat"])
                for ntype in g.ntypes
            }
            self.h = self.rgcn(g, h_dict)

        return super().forward(batch_dict)
