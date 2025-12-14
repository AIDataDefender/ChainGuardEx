import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.function as fn
from dgl.nn.pytorch import GlobalAttentionPooling


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
        num_heads=4,
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
            if self.residual and feat_dst.shape[1] == output.shape[1]:
                output = output + feat_dst
            return output


class CascadedHeteroModel(nn.Module):
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
        super().__init__()
        self.rel_names = rel_names
        self.stage = str(stage)
        self.hidden_dim = hidden_dim

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

        # --- Stage-Specific Heads ---

        if self.stage == "3":
            # Stage 3: Node Classification (Block Level)
            # Output: 8 classes (Multi-label)
            self.classify_ast = nn.Linear(hidden_dim, out_dim)
            self.classify_cfg = nn.Linear(hidden_dim, out_dim)

        else:
            # Stage 1 & 2: Graph/Subgraph Classification (Contract/Function Level)
            # Input: Pre-computed graph embedding [emb_dim]
            # Output: 1 class (Binary: Vuln/Not Vuln)
            emb_dim = 3072  # CodeBERT dimension * 4 heads
            self.classifier = nn.Sequential(
                nn.Linear(emb_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, 1),  # Binary Logit
            )

    def forward(self, batch_dict):
        """
        Forward pass handling both Embedding (Stage 1/2) and Graph (Stage 3) classification.
        """
        if self.stage in ["1", "2"]:
            # Stage 1/2: Direct classification on pre-computed embeddings
            embeddings = batch_dict["embeddings"]  # [batch_size, emb_dim]
            graph_logits = self.classifier(embeddings)
            return {"graph_logits": graph_logits}  # [batch_size, 1]

        else:
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

            h1 = {k: F.elu(v) for k, v in h1.items()}

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

            h2 = {k: F.elu(v) for k, v in h2.items()}

            # --- 2. Node Classification ---
            # Return distinct logits for AST and CFG nodes

            # Safe retrieval for unmapped types
            def get_h(ntype):
                return h2.get(ntype, torch.zeros(0, self.hidden_dim, device=g.device))

            return {
                "ast_logits": self.classify_ast(get_h("ast_node")),
                "cfg_logits": self.classify_cfg(get_h("cfg_node")),
            }
