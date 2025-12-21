import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.function as fn
from dgl.nn.pytorch import GlobalAttentionPooling
from .baseFrameModel import BaseModel


class EdgeGATLayer(nn.Module):
    """
    Custom GAT Layer that fuses Source Node, Destination Node, AND Edge Features
    into the attention mechanism.
    """

    def __init__(
        self,
        in_node_feats_src,
        in_node_feats_dst,
        in_edge_feats,
        out_feats,
        num_heads=2,
        feat_drop=0.1,
        attn_drop=0.1,
    ):
        super(EdgeGATLayer, self).__init__()
        self.num_heads = num_heads
        self.out_feats = out_feats

        # Projections: Map everything to attention space
        self.W_src = nn.Linear(
            in_node_feats_src, num_heads * out_feats, bias=False)
        self.W_dst = nn.Linear(
            in_node_feats_dst, num_heads * out_feats, bias=False)
        self.W_edge = nn.Linear(
            in_edge_feats, num_heads * out_feats, bias=False)

        # Attention Vector: [src || dst || edge] -> 1
        self.attn_vec = nn.Parameter(
            torch.FloatTensor(size=(1, num_heads, 3 * out_feats))
        )

        # Learnable fusion for message
        self.msg_fusion = nn.Linear(2 * out_feats, out_feats)

        self.feat_drop = nn.Dropout(feat_drop)
        self.attn_drop = nn.Dropout(attn_drop)
        self.leaky_relu = nn.LeakyReLU(0.2)

        # Added LayerNorm for better training stability
        self.norm = nn.LayerNorm(out_feats * num_heads)

        self.residual = True

        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain("relu")
        nn.init.xavier_normal_(self.W_src.weight, gain=gain)
        nn.init.xavier_normal_(self.W_dst.weight, gain=gain)
        nn.init.xavier_normal_(self.W_edge.weight, gain=gain)
        nn.init.xavier_normal_(self.attn_vec, gain=gain)
        nn.init.xavier_normal_(self.msg_fusion.weight, gain=gain)

    def forward(self, graph, feat_src, feat_dst, feat_edge):
        with graph.local_scope():
            # 1. Project and Reshape [N, heads, out_feats]
            h_src = self.W_src(self.feat_drop(feat_src)).view(
                -1, self.num_heads, self.out_feats
            )
            h_dst = self.W_dst(self.feat_drop(feat_dst)).view(
                -1, self.num_heads, self.out_feats
            )
            h_edge = self.W_edge(self.feat_drop(feat_edge)).view(
                -1, self.num_heads, self.out_feats
            )

            # 2. Store in graph for message passing
            graph.srcdata.update({"el": h_src})
            graph.dstdata.update({"er": h_dst})
            graph.edata.update({"ee": h_edge})

            # 3. Compute Attention: LeakyReLU(a^T * [h_src || h_dst || h_edge])
            def edge_attention(edges):
                z = torch.cat(
                    [edges.src["el"], edges.dst["er"], edges.data["ee"]], dim=-1
                )
                a = (z * self.attn_vec).sum(dim=-1).unsqueeze(-1)
                return {"e": self.leaky_relu(a)}

            graph.apply_edges(edge_attention)

            # 4. Softmax and Message Passing
            graph.edata["a"] = dgl.ops.edge_softmax(graph, graph.edata["e"])
            graph.edata["a"] = self.attn_drop(graph.edata["a"])

            # Message = Learnable fusion of (Source + Edge_Feature) * Attention_Weight
            def message_func(edges):
                fused = self.msg_fusion(
                    torch.cat([edges.src["el"], edges.data["ee"]], dim=-1))
                return {"m": fused * edges.data["a"]}

            graph.update_all(message_func, fn.sum("m", "ft"))

            # 5. Result: Flatten heads -> [N, heads * out_feats]
            output = graph.dstdata["ft"].flatten(1)
            output = self.norm(output)
            if self.residual and feat_dst.shape[1] == output.shape[1]:
                output = output + feat_dst
            return output


class CascadedHeteroModel(BaseModel):
    def __init__(self, node_dims, edge_dims, hidden_dim, out_dim, rel_names, stage="3"):
        """
        Args:
            node_dims: Dict {'ast_node': 128, 'cfg_node': 128}
            edge_dims: Dict {('ast','call','ast'): 64}
            hidden_dim: Internal feature dimension (e.g. 256)
            out_dim: Number of classes (e.g. 8 for Stage 3, 1 for Stage 1/2)
            rel_names: List of canonical edge types
            stage: "1", "2", or "3" (Controls the head architecture)
        """
        super().__init__(node_dims, edge_dims, hidden_dim, out_dim, rel_names, stage)
        print("="*40, "\n\n")
        print(f"Using EGAT - Stage {self.stage}")
        print("="*40, "\n\n")
        # --- GNN Backbone (Shared across all stages) ---
        # Layer 1
        self.layer1 = nn.ModuleDict()
        for rel in rel_names:
            srctype, etype, dsttype = rel
            rel_key = "_".join(rel)
            self.layer1[rel_key] = EdgeGATLayer(
                in_node_feats_src=node_dims.get(srctype, 128),
                in_node_feats_dst=node_dims.get(dsttype, 128),
                in_edge_feats=edge_dims.get(rel, 64),
                out_feats=hidden_dim // 4,  # 4 heads
                num_heads=4,
            )

        # Layer 2
        self.layer2 = nn.ModuleDict()
        for rel in rel_names:
            rel_key = "_".join(rel)
            self.layer2[rel_key] = EdgeGATLayer(
                in_node_feats_src=hidden_dim,
                in_node_feats_dst=hidden_dim,
                in_edge_feats=edge_dims.get(rel, 64),
                out_feats=hidden_dim // 4,
                num_heads=4,
            )

        # Layer 3 (Added for deeper representations)
        self.layer3 = nn.ModuleDict()
        for rel in rel_names:
            rel_key = "_".join(rel)
            self.layer3[rel_key] = EdgeGATLayer(
                in_node_feats_src=hidden_dim,
                in_node_feats_dst=hidden_dim,
                in_edge_feats=edge_dims.get(rel, 64),
                out_feats=hidden_dim // 4,
                num_heads=4,
            )

        # --- Stage-Specific Heads ---
        if self.stage == "3":
            # Redefine with deeper MLPs
            self.classify_ast = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.LayerNorm(hidden_dim // 2),
                nn.GELU(),
                nn.Dropout(0.3),
                nn.Linear(hidden_dim // 2, hidden_dim // 4),
                nn.LayerNorm(hidden_dim // 4),
                nn.GELU(),
                nn.Dropout(0.2),
                nn.Linear(hidden_dim // 4, out_dim)
            )
            self.classify_cfg = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.LayerNorm(hidden_dim // 2),
                nn.GELU(),
                nn.Dropout(0.3),
                nn.Linear(hidden_dim // 2, hidden_dim // 4),
                nn.LayerNorm(hidden_dim // 4),
                nn.GELU(),
                nn.Dropout(0.2),
                nn.Linear(hidden_dim // 4, out_dim)
            )
        else:
            # Attention pooling gates for AST and CFG
            self.ast_pool_gate = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 4),
                nn.GELU(),
                nn.Linear(hidden_dim // 4, 1),
                nn.LeakyReLU()
            )
            self.cfg_pool_gate = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 4),
                nn.GELU(),
                nn.Linear(hidden_dim // 4, 1),
                nn.LeakyReLU()
            )
            self.attention_temp = nn.Parameter(
                torch.tensor(1.0))  # Learnable temperature

            # Redefine deeper classifier
            self.classifier = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.LayerNorm(hidden_dim // 2),
                nn.GELU(),
                nn.Dropout(0.05),
                nn.Linear(hidden_dim // 2, hidden_dim // 4),
                nn.GELU(),
                nn.Linear(hidden_dim // 4, 1),  # Binary Logit
            )

    def forward(self, batch_dict):
        """
        Forward pass handling both Embedding (Stage 1/2) and Graph (Stage 3) classification.
        """

        # Stage 3: GNN on graph
        g = batch_dict["graph"]

        # In heterogeneous graphs, features are stored per node/edge type
        # g.ndata["feat"] returns a dict: {node_type: tensor}
        # g.edata["feat"] returns a dict: {edge_type_tuple: tensor}

        # --- 1. GNN Backbone (Message Passing) ---

        # Layer 1
        h1 = {
            ntype: torch.zeros(g.num_nodes(
                ntype), self.hidden_dim, device=g.device)
            for ntype in g.ntypes
        }

        for rel in self.rel_names:
            srctype, etype, dsttype = rel
            rel_key = "_".join(rel)

            # Skip if this edge type doesn't exist in the current graph batch
            if rel not in g.canonical_etypes or g.num_edges(rel) == 0:
                continue

            # Extract features for this relation
            src_feat = g.nodes[srctype].data["feat"]
            dst_feat = g.nodes[dsttype].data["feat"]
            edge_feat = g.edges[rel].data["feat"]

            res = self.layer1[rel_key](
                g[rel], src_feat, dst_feat, edge_feat
            )

            h1[dsttype] += res

        h1 = {k: F.gelu(v) for k, v in h1.items()}

        # Layer 2
        h2 = {ntype: torch.zeros_like(h1[ntype]) for ntype in g.ntypes}

        for rel in self.rel_names:
            srctype, etype, dsttype = rel
            rel_key = "_".join(rel)

            # Skip if this edge type doesn't exist in the current graph batch
            if rel not in g.canonical_etypes or g.num_edges(rel) == 0:
                continue

            # Extract edge features for this relation
            edge_feat = g.edges[rel].data["feat"]

            res = self.layer2[rel_key](
                g[rel], h1[srctype], h1[dsttype], edge_feat)
            h2[dsttype] += res

        h2 = {k: F.gelu(v) for k, v in h2.items()}

        # Layer 3 (Added for deeper representations)
        h3 = {ntype: torch.zeros_like(h2[ntype]) for ntype in g.ntypes}

        for rel in self.rel_names:
            srctype, etype, dsttype = rel
            rel_key = "_".join(rel)

            # Skip if this edge type doesn't exist in the current graph batch
            if rel not in g.canonical_etypes or g.num_edges(rel) == 0:
                continue

            # Extract edge features for this relation
            edge_feat = g.edges[rel].data["feat"]

            res = self.layer3[rel_key](
                g[rel], h2[srctype], h2[dsttype], edge_feat)
            h3[dsttype] += res

        h3 = {k: F.gelu(v) for k, v in h3.items()}

        # Skip connection from Layer 1 to Layer 3 for better gradient flow
        h3 = {k: v + h1[k] for k, v in h3.items()}

        # Set self.h for base class
        self.h = h3

        if self.stage == "3":
            return super().forward(batch_dict)
        else:
            # Attention-weighted pooling for graph-level representations
            def attention_pool(node_feats, gate_nn, g, ntype):
                """Apply attention-based pooling to node features."""
                if g.num_nodes(ntype) == 0:
                    return torch.zeros(g.batch_size, self.hidden_dim, device=g.device)

                # Compute attention scores
                attn_scores = gate_nn(node_feats)  # [N, 1]

                # Store in graph for batch-wise pooling
                g.nodes[ntype].data['h'] = node_feats
                g.nodes[ntype].data['a'] = attn_scores / \
                    self.attention_temp  # Apply temperature

                # Softmax attention per graph in batch
                g.nodes[ntype].data['a'] = dgl.softmax_nodes(
                    g, 'a', ntype=ntype)

                # Weighted sum
                g.nodes[ntype].data['h_weighted'] = node_feats * \
                    g.nodes[ntype].data['a']
                pooled = dgl.readout_nodes(
                    g, 'h_weighted', ntype=ntype, op='sum')

                return pooled

            ast_feats = self.h.get('ast_node', torch.zeros(
                0, self.hidden_dim, device=g.device))
            cfg_feats = self.h.get('cfg_node', torch.zeros(
                0, self.hidden_dim, device=g.device))

            ast_pooled = attention_pool(
                ast_feats, self.ast_pool_gate, g, 'ast_node')
            cfg_pooled = attention_pool(
                cfg_feats, self.cfg_pool_gate, g, 'cfg_node')

            # Concatenate pooled features with max pooling for robustness
            # Shape: (batch_size, hidden_dim * 2)
            pooled_features = torch.cat([ast_pooled, cfg_pooled], dim=-1)

            # Shape: (batch_size, 1) - one logit per graph
            return {"graph_logits": self.classifier(pooled_features)}
