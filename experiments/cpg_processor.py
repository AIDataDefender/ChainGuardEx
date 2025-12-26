from typing import Any, Dict, List, Set
import torch
import torch.multiprocessing as torch_mp
import torch.nn as nn
import dgl
import networkx as nx
import numpy as np
from collections import defaultdict
from tqdm import tqdm
from the_utils.c_2constants import GraphAttributes, EdgeTypes, NodeTypes
import json
import sys
import os
from pathlib import Path
from the_utils import graph_utils
import traceback
import multiprocessing
import concurrent.futures
from functools import partial

# Ensure torch uses file-system sharing to avoid fd/ancdata issues in multiprocessing
torch_mp.set_sharing_strategy("file_system")


def process_project_worker(project_item, vuln_lookup, device, batch_size, checkpoint_dir, embedding_model_conf):
    p_name, nx_g = project_item
    vuln_data = vuln_lookup.get(p_name)
    if not vuln_data:
        return None

    # Load model in worker
    from transformers import AutoModel, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        embedding_model_conf.name, use_fast=True)
    model = AutoModel.from_pretrained(embedding_model_conf.name)
    model.eval()
    model.to(device)

    # Create processor
    processor = CPG_Processor(tokenizer=tokenizer, model=model, device=device, batch_size=batch_size,
                              checkpoint_dir=checkpoint_dir, embedding_model_conf=embedding_model_conf)

    # Process (checkpoint is saved inside the processor)
    result = processor._process_single_project(p_name, nx_g, vuln_data, return_results=False)

    if result:
        # Cleanup to free memory
        del processor, model, tokenizer
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return p_name
    # Cleanup to free memory
    del processor, model, tokenizer
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return None


class EmbeddingModel(nn.Module):
    def __init__(
        self,
        tokenizer,
        model,
        device: str,
        max_length: int = 256,
    ):
        """
        Simplified Embedding Model:
        - Runs CodeBERT for text.
        - Returns raw embeddings (no random projections).
        - Structural features are handled by the GNN later.
        """
        super().__init__()
        self.tokenizer = tokenizer
        self.model = model
        self.device = device
        self.max_length = max_length

    def forward(self, texts: List[str]) -> torch.Tensor:
        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        ).to(self.device)

        with torch.no_grad():
            out = self.model(**inputs)
            # Use CLS token embedding
            bert_emb = out.last_hidden_state[:, 0, :]

        return bert_emb.detach().cpu()


class CPG_Processor:
    def __init__(
        self,
        tokenizer=None,
        model=None,
        device="cuda" if torch.cuda.is_available() else "cpu",
        batch_size=32,  # Increased from 16
        checkpoint_dir="/mnt/d/KLTN2/DatasetEtherScanio/EtherScanio/checkpoints/processed_graphs",
        embedding_model_conf=None,
    ):
        self.device = device
        self.batch_size = batch_size
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        print("="*40,"\n\n", "CHECKPOINT DIR:", self.checkpoint_dir,"\n","="*40)
        self.embedding_model_conf = embedding_model_conf
        self.embedding_model = None
        if tokenizer or model:
            print(f"Loading CodeBERT on {device}...")
            self.tokenizer = tokenizer
            self.model = model
            self.model.eval()
            # Initialize Embedding Helper
            self.embedding_model = EmbeddingModel(
                tokenizer=self.tokenizer, model=self.model, device=self.device
            ).to(self.device)

        # Feature definitions
        self.bool_fields = [
            "expression_has_require",
            "expression_has_revert",
            "expression_is_call_like",
            "node_is_branch_like",
            "node_is_terminator",
            "node_is_callsite",
        ]
        self.numeric_fields = [
            "ir_line_count",
            "ir_instruction_count",
            "ir_unique_var_count",
            "ir_edge_count",
            "ir_call_count",
            "ir_assign_count",
            "ir_phi_count",
            "ir_return_count",
            "ir_condition_count",
            "ir_tmp_ref_count",
            "ir_def_count",
            "ir_use_count",
        ]
        self.cfg_node_label = NodeTypes.CFG_NODE.value.lower()
        self.ast_node_label = NodeTypes.AST_NODE.value.lower()

        self.embedding_dims = {
            "cfg_node": embedding_model_conf.embedding_dim + self._compute_struct_feature_size("cfg"),
            "ast_node": embedding_model_conf.embedding_dim + self._compute_struct_feature_size("ast"),
            "edge": embedding_model_conf.embedding_dim,
        }

    def _normalize_node_type(self, raw: Any) -> str:
        """Return canonical node type string: 'cfg_node' or 'ast_node'."""
        try:
            if hasattr(raw, "value"):
                raw = raw.value
            s = str(raw).strip().lower()
        except Exception:
            return self.cfg_node_label

        # Strip enum-like prefixes if present
        if s.startswith("nodetypes."):
            s = s.split(".", 1)[1]

        # Common variants
        mapping = {
            "cfg_node": self.cfg_node_label,
            "cfg": self.cfg_node_label,
            "ast_node": self.ast_node_label,
            "ast": self.ast_node_label,
            # Legacy variants are no longer supported; prefer canonical names only.
        }
        if s in mapping:
            return mapping[s]
        if "ast" in s:
            return self.ast_node_label
        if "cfg" in s:
            return self.cfg_node_label
        return self.cfg_node_label

    def _validate_project_stage_outputs(
        self,
        project_name: str,
        stage: int,
        graphs: Dict[str, dgl.DGLGraph],
        labels: Dict[str, Any],
    ) -> None:
        """Fail-fast validation to prevent schema drift between processor and dataset."""
        if not isinstance(graphs, dict) or not isinstance(labels, dict):
            raise TypeError(
                f"{project_name} stage{stage}: graphs/labels must be dicts, got graphs={type(graphs)} labels={type(labels)}"
            )

        g_keys = set(graphs.keys())
        l_keys = set(labels.keys())
        if g_keys != l_keys:
            missing = sorted(list(g_keys - l_keys))[:10]
            extra = sorted(list(l_keys - g_keys))[:10]
            raise ValueError(
                f"{project_name} stage{stage}: graph/label keys mismatch graphs={len(g_keys)} labels={len(l_keys)} missing={missing} extra={extra}"
            )

        # Validate stage 3 label lengths match node counts (when node types exist)
        if stage == 3:
            for subkey in list(g_keys)[:50]:  # cap to keep runtime reasonable
                g = graphs[subkey]
                lbl = labels[subkey]
                if not isinstance(lbl, dict):
                    raise TypeError(
                        f"{project_name} stage3 {subkey}: label must be dict, got {type(lbl)}"
                    )
                if "cfg_node" not in lbl or "ast_node" not in lbl:
                    raise ValueError(
                        f"{project_name} stage3 {subkey}: label must contain cfg_node/ast_node"
                    )

                cfg_nodes = g.num_nodes(
                    self.cfg_node_label) if self.cfg_node_label in g.ntypes else 0
                ast_nodes = g.num_nodes(
                    self.ast_node_label) if self.ast_node_label in g.ntypes else 0
                cfg_labels = lbl.get("cfg_node") or []
                ast_labels = lbl.get("ast_node") or []
                if len(cfg_labels) != cfg_nodes:
                    raise ValueError(
                        f"{project_name} stage3 {subkey}: cfg_node labels {len(cfg_labels)} != cfg_node nodes {cfg_nodes}"
                    )
                if len(ast_labels) != ast_nodes:
                    raise ValueError(
                        f"{project_name} stage3 {subkey}: ast_node labels {len(ast_labels)} != ast_node nodes {ast_nodes}"
                    )

                # HARD: label vectors must be multi-hot of fixed width with only 0/1 values.
                expected_dim = len(getattr(graph_utils, "OWASP_VULN", []))
                if expected_dim <= 0:
                    raise RuntimeError("OWASP_VULN is not defined or empty")

                def _check_rows(rows, node_type: str):
                    for i, row in enumerate(rows[:5]):  # sample first few rows
                        if not isinstance(row, (list, tuple)):
                            raise TypeError(
                                f"{project_name} stage3 {subkey}: {node_type}[{i}] must be list/tuple, got {type(row)}"
                            )
                        if len(row) != expected_dim:
                            raise ValueError(
                                f"{project_name} stage3 {subkey}: {node_type}[{i}] dim {len(row)} != {expected_dim}"
                            )
                        bad = [v for v in row if v not in (0, 1, 0.0, 1.0)]
                        if bad:
                            raise ValueError(
                                f"{project_name} stage3 {subkey}: {node_type}[{i}] contains non-binary values (sample={bad[:5]})"
                            )

                _check_rows(cfg_labels, "cfg_node")
                _check_rows(ast_labels, "ast_node")

        if stage in (1, 2):
            for subkey in list(g_keys)[:50]:
                v = labels[subkey]
                if isinstance(v, bool):
                    v = int(v)
                if not isinstance(v, int):
                    raise TypeError(
                        f"{project_name} stage{stage} {subkey}: label must be int 0/1, got {type(v)}"
                    )
                if int(v) not in (0, 1):
                    raise ValueError(
                        f"{project_name} stage{stage} {subkey}: label must be 0/1, got {v}"
                    )

    def _compute_struct_feature_size(self, type="cfg") -> int:
        # Helper to know input dim for GNN
        if type == "cfg":
            return (
                2
                + len(graph_utils.CFG_NODE_TYPE_LIST)
                + len(self.bool_fields)
                + len(self.numeric_fields)
            )
        else:
            return (
                2
                + len(graph_utils.AST_NODE_TYPE_LIST)
                + len(graph_utils.VAR_VISIBILITY)
                + len(graph_utils.FUNC_VISIBILITY)
                + len(graph_utils.VAR_STORAGE)
                + len(graph_utils.STATE_MUTABILITY)
                + len(graph_utils.EVENT_META)
                + len(graph_utils.CONTRACT_KIND)
            )

    def _get_bert_embeddings(self, text_list: List[str]) -> torch.Tensor:
        """Batched BERT embedding generation."""
        if not text_list:
            return torch.zeros((0, 768), dtype=torch.float32)

        embeddings = []
        for i in range(0, len(text_list), self.batch_size):
            batch_text = text_list[i: i + self.batch_size]
            # Clean text
            batch_text = [
                str(t) if t and str(t).strip() else "[UNK]" for t in batch_text
            ]

            emb = self.embedding_model(batch_text)  # type: ignore
            embeddings.append(emb)

        return torch.cat(embeddings, dim=0)

    def _build_node_struct_features(
        self, node_type: str, data: Dict[str, Any]
    ) -> List[float]:
        """Extracts numerical/boolean features from node attributes."""
        feats: List[float] = []
        normalized_type = node_type.lower()

        # 1. Base Node Type (One-Hot)
        node_type_one_hot = (
            [1.0, 0.0] if normalized_type == self.cfg_node_label else [0.0, 1.0]
        )
        feats.extend(node_type_one_hot)

        # 2. Sub Node Type (One-Hot)
        sub_type = str(
            data.get(GraphAttributes.SUB_NODE_TYPE, "") or "").lower()

        if normalized_type == self.cfg_node_label:
            # CFG Specific
            feats.extend(
                graph_utils.node_type_to_one_hot(
                    sub_type, graph_utils.CFG_NODE_TYPE_LIST
                )
            )
            for field in self.bool_fields:
                feats.append(1.0 if data.get(field) else 0.0)
            for field in self.numeric_fields:
                val = data.get(field, 0)
                try:
                    feats.append(float(val))
                except Exception:
                    feats.append(0.0)
        else:
            # AST Specific
            feats.extend(
                graph_utils.node_type_to_one_hot(
                    sub_type, graph_utils.AST_NODE_TYPE_LIST
                )
            )

            # Helper to safely get one-hot
            def get_vec(key, domain, default="none"):
                val = str(data.get(key, default)).lower()
                return graph_utils.node_type_to_one_hot(val, domain)

            feats.extend(
                get_vec("visibility", graph_utils.VAR_VISIBILITY))  # Var Vis
            # Func Vis
            feats.extend(get_vec("visibility", graph_utils.FUNC_VISIBILITY))
            feats.extend(get_vec("storage_location", graph_utils.VAR_STORAGE))
            feats.extend(get_vec("state_mutability",
                                 graph_utils.STATE_MUTABILITY))

            # Event Meta (simplified)
            event_vec = [0.0] * len(graph_utils.EVENT_META)
            if data.get("anonymous"):
                event_vec[0] = 1.0  # simplistic mapping
            feats.extend(event_vec)

            feats.extend(get_vec("contract_kind", graph_utils.CONTRACT_KIND))

        return feats

    # =========================================================================
    # HIERARCHY & LABELING LOGIC (The "Bubble Up")
    # =========================================================================

    def _build_hierarchy_map(self, nx_graph: nx.DiGraph) -> Dict[str, Any]:
        """Maps nodes to File -> Contract -> Function hierarchy."""
        hierarchy = {}
        temp_nodes = {}  # Collect nodes with missing attributes
        nid_to_contract = {}  # nid -> (file, contract)

        for nid, data in nx_graph.nodes(data=True):
            n_type = self._normalize_node_type(
                data.get(GraphAttributes.NODE_TYPE, NodeTypes.CFG_NODE)
            )
            contract = str(data.get(GraphAttributes.CONTRACT, "")).strip()
            function = str(data.get(GraphAttributes.FUNCTION, "")).strip()
            file = str(data.get(GraphAttributes.FILE, "")).strip()

            if not contract or not file or not function:
                temp_nodes[nid] = data
                continue  # Collect in temp

            # 1. Ensure File Bucket Exists
            if file not in hierarchy:
                hierarchy[file] = {}

            # 2. Ensure Contract Bucket Exists within File
            if contract not in hierarchy[file]:
                hierarchy[file][contract] = {
                    "cfg_nodes": set(),
                    "ast_nodes": set(),
                    "functions": {},
                }

            # Normalize types
            is_cfg = n_type == self.cfg_node_label
            is_ast = n_type == self.ast_node_label

            # 3. Determine Target (Function or Contract level)
            if function and function != "None":
                # Ensure Function Bucket Exists
                if function not in hierarchy[file][contract]["functions"]:
                    hierarchy[file][contract]["functions"][function] = {
                        "cfg_nodes": set(),
                        "ast_nodes": set(),
                    }
                target = hierarchy[file][contract]["functions"][function]
            else:
                target = hierarchy[file][contract]  # Contract-level node

            # 4. Append to Target
            if is_cfg:
                target["cfg_nodes"].add(nid)
            elif is_ast:
                target["ast_nodes"].add(nid)

            # Record contract for this nid
            nid_to_contract[nid] = (file, contract)

        # Now, incorporate temp nodes via edges
        for u, v in nx_graph.edges():
            if u in nid_to_contract and v in temp_nodes:
                file, contract = nid_to_contract[u]
                n_type = str(
                    temp_nodes[v].get(
                        GraphAttributes.NODE_TYPE, NodeTypes.CFG_NODE)
                ).lower()
                n_type = self._normalize_node_type(n_type)
                if n_type == self.cfg_node_label:
                    hierarchy[file][contract]["cfg_nodes"].add(v)
                elif n_type == self.ast_node_label:
                    hierarchy[file][contract]["ast_nodes"].add(v)
                nid_to_contract[v] = (file, contract)  # Add to map
                # no remove as there could be multiple edges
            elif v in nid_to_contract and u in temp_nodes:
                file, contract = nid_to_contract[v]
                n_type = str(
                    temp_nodes[u].get(
                        GraphAttributes.NODE_TYPE, NodeTypes.CFG_NODE)
                ).lower()
                n_type = self._normalize_node_type(n_type)
                if n_type == self.cfg_node_label:
                    hierarchy[file][contract]["cfg_nodes"].add(u)
                elif n_type == self.ast_node_label:
                    hierarchy[file][contract]["ast_nodes"].add(u)
                nid_to_contract[u] = (file, contract)
                # no remove as there could be multiple edges

        # Validation: Check if all nodes are accounted for
        total_hierarchy_nodes = 0
        for file_data in hierarchy.values():
            for contract_data in file_data.values():
                total_hierarchy_nodes += len(contract_data["cfg_nodes"]) + len(
                    contract_data["ast_nodes"]
                )
                for func_data in contract_data["functions"].values():
                    total_hierarchy_nodes += len(func_data["cfg_nodes"]) + len(
                        func_data["ast_nodes"]
                    )

        graph_node_count = len(nx_graph.nodes())
        remaining_temp = len(temp_nodes)
        if total_hierarchy_nodes != graph_node_count:
            print(
                f"Warning: Hierarchy nodes ({total_hierarchy_nodes}) != Graph nodes ({graph_node_count})"
            )
            print(f"Remaining temp nodes: {remaining_temp}")
        return hierarchy

    def _create_stage_labels(self, nx_graph, hierarchy, vuln_data):
        """Generates labels for all 3 stages simultaneously."""
        # Build Vulnerability Map from vuln_data
        vuln_map = {"ast": {}, "cfg": {}}
        # Iterate through each vulnerability entry
        for vuln_key, vuln_entry in vuln_data.items():
            vuln_label = str(vuln_entry.get("owasp_id", "")).upper().strip()
            if not vuln_label:
                raise ValueError(f"Missing owasp_id in vuln entry {vuln_key}")
            if vuln_label not in graph_utils.OWASP_VULN and vuln_label not in graph_utils.OWASP_VULN2:
                raise ValueError(
                    f"Unknown owasp_id '{vuln_label}' in vuln entry {vuln_key}; not in OWASP_VULN"
                )

            # Extract AST nodes
            for nid, node_data in vuln_entry.get("ast_nodes", {}).items():
                nid = str(nid)
                if nid not in vuln_map["ast"]:
                    vuln_map["ast"][nid] = []
                if vuln_label in vuln_map["ast"][nid]:
                    continue  # skip same label
                vuln_map["ast"][nid].append(vuln_label)

            # Extract CFG nodes
            for nid, node_data in vuln_entry.get(
                "nodes", {}
            ).items():  # yep - cfg_node is "nodes"
                nid = str(nid)
                if nid not in vuln_map["cfg"]:
                    vuln_map["cfg"][nid] = []
                if vuln_label in vuln_map["cfg"][nid]:
                    continue  # skip same label
                vuln_map["cfg"][nid].append(vuln_label)

        # Stage 3 (Block)
        stage3_labels = defaultdict(list)
        node_is_vuln = {}  # Cache for bubble up
        alert_dict = defaultdict(lambda: {})

        print("\n=== Stage Label Creation Debug ===")
        print(
            f"Input vuln_map: AST nodes={list(vuln_map['ast'].keys())}, CFG nodes={list(vuln_map['cfg'].keys())}"
        )

        for nid, data in nx_graph.nodes(data=True):
            n_type = str(
                data.get(GraphAttributes.NODE_TYPE, NodeTypes.CFG_NODE)
            ).lower()

            owasp_list = vuln_map["ast"].get(str(nid), []) + vuln_map["cfg"].get(
                str(nid), []
            )
            label = graph_utils.vuln_to_label(owasp_list)

            # HARD: vuln list present but no known OWASP match.
            if owasp_list and not any(v > 0 for v in label):
                raise ValueError(
                    f"Node {nid} has OWASP list {owasp_list} but produced all-zero label (unknown IDs?)"
                )

            # Check for duplicate nid with different label
            if nid in alert_dict[n_type]:
                if alert_dict[n_type][nid] != label:
                    print(
                        f"Alert: Node {nid} in {n_type} has different labels: {alert_dict[n_type][nid]} vs {label}"
                    )
            alert_dict[n_type][nid] = label
            stage3_labels[n_type].append(label)

            node_is_vuln[nid] = any(
                label_value > 0 for label_value in label)  # max-pooling
            if node_is_vuln[nid]:
                print(
                    f"Node {nid} {n_type} marked as vulnerable with labels: {owasp_list}"
                )

        print(
            f"Total vulnerable nodes in stage3: {sum(1 for v in node_is_vuln.values() if v)}"
        )

        # Stage 2 (Function)
        stage2_labels = {}
        for file_name, file_data in hierarchy.items():
            for contract, c_data in file_data.items():
                for func, f_data in c_data.get("functions", {}).items():
                    all_nodes = list(f_data["cfg_nodes"]) + \
                        list(f_data["ast_nodes"])
                    is_vuln = any(node_is_vuln.get(n, False)
                                  for n in all_nodes)
                    # Key by the pruned stage2 graph subkey to avoid collisions across files.
                    stage2_labels[f"{file_name}_{contract}_{func}"] = 1 if is_vuln else 0
                    if is_vuln:
                        print(
                            f"Function {file_name}_{contract}_{func} marked vulnerable")

        # Stage 1 (Contract)
        stage1_labels = {}
        for file_name, file_data in hierarchy.items():
            for contract, c_data in file_data.items():
                # Check functions
                func_vuln = any(
                    stage2_labels.get(f"{file_name}_{contract}_{func}", 0) == 1
                    for func in c_data.get("functions", {})
                )
                # Check contract-level nodes (rare but possible)
                node_vuln = any(
                    node_is_vuln.get(n, False)
                    for n in list(c_data.get("cfg_nodes", set()))
                    + list(c_data.get("ast_nodes", set()))
                )
                contract_key = f"{file_name}_{contract}"
                stage1_labels[contract_key] = 1 if (
                    func_vuln or node_vuln) else 0
                if func_vuln or node_vuln:
                    print(
                        f"Contract {contract_key} marked vulnerable (func_vuln={func_vuln}, node_vuln={node_vuln})"
                    )

        vuln_count = sum(1 for v in node_is_vuln.values() if v)
        return stage3_labels, stage2_labels, stage1_labels, vuln_count, vuln_map

    # =========================================================================
    # PRUNING LOGIC (The "Zoom In")
    # =========================================================================
    def _prune_stage1(self, nx_graph, stage1_labels, hierarchy, vuln_map, n_hop=1):
        """Split big graph into smaller graphs per contract."""
        result = {}
        for file_name, file_data in hierarchy.items():
            for contract_name, contract_data in file_data.items():
                # Get all nodes for this contract
                keep_nodes = set()
                keep_nodes.update(contract_data.get("cfg_nodes", set()))
                keep_nodes.update(contract_data.get("ast_nodes", set()))

                # Add function nodes for this contract
                for f_data in contract_data.get("functions", {}).values():
                    keep_nodes.update(f_data.get("cfg_nodes", set()))
                    keep_nodes.update(f_data.get("ast_nodes", set()))

                # Expand to n-hop neighbors to preserve context
                expanded_nodes = set(keep_nodes)
                for _ in range(n_hop):
                    new_neighbors = set()
                    for node in expanded_nodes:
                        if node in nx_graph:
                            new_neighbors.update(nx_graph.neighbors(node))
                    expanded_nodes.update(new_neighbors)
                keep_nodes = expanded_nodes

                # Create subgraph for this contract
                if keep_nodes:
                    contract_graph = nx_graph.subgraph(keep_nodes).copy()

                    # prune if node < 10:
                    if (
                        contract_graph.number_of_nodes() < 2
                        or contract_graph.number_of_edges() < 1
                    ):
                        # Check if vulnerable nodes are present
                        has_vuln = any(
                            str(nid) in vuln_map["ast"] or str(
                                nid) in vuln_map["cfg"]
                            for nid in contract_graph.nodes()
                        )
                        if has_vuln:
                            print(
                                f"REPORT: Discarding vulnerable contract graph: {file_name}_{contract_name} "
                                f"with {contract_graph.number_of_nodes()} nodes, {contract_graph.number_of_edges()} edges. "
                                f"Vulnerable nodes present but graph too small."
                            )
                        # else:
                        #     print(
                        #         f"Skipping small contract graph: {file_name}_{contract_name} with {contract_graph.number_of_nodes()} nodes and {contract_graph.number_of_edges()} edges"
                        #     )
                        continue

                    # Get label for this contract
                    contract_key = f"{file_name}_{contract_name}"
                    contract_label = stage1_labels.get(contract_key, 0)

                    # Create key as "file_contract1", "file_contract2", etc.
                    result[contract_key] = {
                        "graph": contract_graph,
                        "label": contract_label,
                    }
        return result

    def _prune_stage2(self, stage1_result, stage2_labels, hierarchy, vuln_map, n_hop=1):
        """Split big graph into smaller graphs per function."""
        result = {}
        for file_name, file_data in hierarchy.items():
            for contract_name, contract_data in file_data.items():
                contract_key = f"{file_name}_{contract_name}"
                if contract_key not in stage1_result:
                    continue
                contract_graph = stage1_result[contract_key]["graph"]

                # Contract-level nodes (shared context)
                contract_nodes = set()
                contract_nodes.update(contract_data.get("cfg_nodes", set()))
                contract_nodes.update(contract_data.get("ast_nodes", set()))

                for func_name, func_data in contract_data.get("functions", {}).items():
                    # Get all nodes for this function
                    # Include contract context
                    keep_nodes = set(contract_nodes)
                    keep_nodes.update(func_data.get("cfg_nodes", set()))
                    keep_nodes.update(func_data.get("ast_nodes", set()))

                    # Expand to n-hop neighbors to preserve context
                    expanded_nodes = set(keep_nodes)
                    for _ in range(n_hop):
                        new_neighbors = set()
                        for node in expanded_nodes:
                            if node in contract_graph:
                                new_neighbors.update(
                                    contract_graph.neighbors(node))
                        expanded_nodes.update(new_neighbors)
                    keep_nodes = expanded_nodes

                    # Create subgraph for this function
                    if keep_nodes:
                        func_graph = contract_graph.subgraph(keep_nodes).copy()
                        # prune if node < 3:
                        if (
                            func_graph.number_of_nodes() < 2
                            or func_graph.number_of_edges() < 1
                        ):
                            # Check if vulnerable nodes are present
                            has_vuln = any(
                                str(nid) in vuln_map["ast"]
                                or str(nid) in vuln_map["cfg"]
                                for nid in func_graph.nodes()
                            )
                            if has_vuln:
                                print(
                                    f"REPORT: Discarding vulnerable function graph: {file_name}_{contract_name}_{func_name} "
                                    f"with {func_graph.number_of_nodes()} nodes, {func_graph.number_of_edges()} edges. "
                                    f"Vulnerable nodes present but graph too small."
                                )
                            # else:
                            #     print(
                            #         f"Skipping small function graph: {file_name}_{contract_name}_{func_name} with {func_graph.number_of_nodes()} nodes and {func_graph.number_of_edges()} edges"
                            #     )
                            continue
                        # Get label for this function
                        # Create key as "file_contract_func"
                        result_key = f"{file_name}_{contract_name}_{func_name}"
                        func_label = stage2_labels.get(result_key, 0)
                        result[result_key] = {
                            "graph": func_graph, "label": func_label}
        return result

    def _prune_stage3(self, stage2_result, vuln_data, max_negative_functions=2):
        """Keep vulnerable blocks + Dataflow/Control Context for vulnerable functions.

        Also includes up to max_negative_functions non-vulnerable functions for training balance.
        """
        # Build Vulnerability Map from vuln_data
        vuln_map = {"ast": {}, "cfg": {}}
        # Iterate through each vulnerability entry
        for vuln_key, vuln_entry in vuln_data.items():
            # Extract AST nodes
            for nid, node_data in vuln_entry.get("ast_nodes", {}).items():
                vuln_label = vuln_entry["owasp_id"]
                if nid not in vuln_map["ast"]:
                    vuln_map["ast"][nid] = []
                if vuln_label in vuln_map["ast"][nid]:
                    continue  # skip same label
                vuln_map["ast"][nid].append(vuln_label)

            # Extract CFG nodes
            for nid, node_data in vuln_entry.get(
                "nodes", {}
            ).items():  # yep - cfg_node is "nodes"
                vuln_label = vuln_entry["owasp_id"]
                if nid not in vuln_map["cfg"]:
                    vuln_map["cfg"][nid] = []
                if vuln_label in vuln_map["cfg"][nid]:
                    continue  # skip same label
                vuln_map["cfg"][nid].append(vuln_label)

        result = {}
        negative_count = 0

        # First, process all vulnerable functions
        for key, data in stage2_result.items():
            if data["label"] == 1:  # Process vulnerable functions
                func_graph = data["graph"]
                func_nodes = set(str(n) for n in func_graph.nodes())
                # Filter vuln_map to only include nodes present in the func_graph
                filtered_vuln_map = {
                    "ast": {
                        nid: vulns
                        for nid, vulns in vuln_map["ast"].items()
                        if nid in func_nodes
                    },
                    "cfg": {
                        nid: vulns
                        for nid, vulns in vuln_map["cfg"].items()
                        if nid in func_nodes
                    },
                }

                # Convert vuln_map to indexed labels by iterating nodes in same order as nx_to_dgl
                indexed_labels = {"cfg_node": [], "ast_node": []}
                for nid, node_data in func_graph.nodes(data=True):
                    n_type = self._normalize_node_type(
                        node_data.get(GraphAttributes.NODE_TYPE,
                                      NodeTypes.CFG_NODE)
                    )

                    # Get OWASP list for this node
                    if n_type == self.cfg_node_label:
                        owasp_list = filtered_vuln_map.get(
                            "cfg", {}).get(str(nid), [])
                        label = graph_utils.vuln_to_label(owasp_list)
                        indexed_labels["cfg_node"].append(label)
                    elif n_type == self.ast_node_label:
                        owasp_list = filtered_vuln_map.get(
                            "ast", {}).get(str(nid), [])
                        label = graph_utils.vuln_to_label(owasp_list)
                        indexed_labels["ast_node"].append(label)

                result[key] = {"graph": func_graph, "label": indexed_labels}

        # Then, add some non-vulnerable functions for training balance
        for key, data in stage2_result.items():
            # Process non-vulnerable functions
            if data["label"] == 0 and negative_count < max_negative_functions:
                func_graph = data["graph"]
                # For non-vulnerable functions, all node labels are zeros
                indexed_labels = {"cfg_node": [], "ast_node": []}
                for nid, node_data in func_graph.nodes(data=True):
                    n_type = self._normalize_node_type(
                        node_data.get(GraphAttributes.NODE_TYPE,
                                      NodeTypes.CFG_NODE)
                    )

                    # All labels are zeros for non-vulnerable functions
                    label = [0] * len(graph_utils.OWASP_VULN)
                    if n_type == self.cfg_node_label:
                        indexed_labels["cfg_node"].append(label)
                    elif n_type == self.ast_node_label:
                        indexed_labels["ast_node"].append(label)

                result[key] = {"graph": func_graph, "label": indexed_labels}
                negative_count += 1

        return result

    # =========================================================================
    # DGL CONVERSION (Generic)
    # =========================================================================

    def nx_to_dgl(self, nx_graph):
        """
        Generic converter for any stage.
        """
        try:
            # print(f"Input graph has {nx_graph.number_of_nodes()} nodes, {nx_graph.number_of_edges()} edges")
            # 1. Map Global IDs to Local IDs per Type
            node_storage = defaultdict(
                lambda: {"texts": [], "structs": [], "ids": []})
            global_to_local = {}

            for nid, data in nx_graph.nodes(data=True):
                n_type = self._normalize_node_type(
                    data.get(GraphAttributes.NODE_TYPE, NodeTypes.CFG_NODE)
                )

                # Text Feature
                if n_type == self.ast_node_label:
                    text = f"{data.get(GraphAttributes.LABEL, '')} {data.get(GraphAttributes.SUB_NODE_TYPE, '')}"
                else:
                    text = f"{data.get(GraphAttributes.EXPRESSION, '')} {data.get(GraphAttributes.IR, '')}"

                # Struct Feature
                struct = self._build_node_struct_features(n_type, data)

                idx = len(node_storage[n_type]["ids"])
                global_to_local[nid] = (n_type, idx)

                node_storage[n_type]["texts"].append(text)
                node_storage[n_type]["structs"].append(struct)
                node_storage[n_type]["ids"].append(nid)

                # Store contract and function IDs for Stage 1/2 pooling
                contract = str(
                    data.get(GraphAttributes.CONTRACT, "UnknownContract")
                ).strip()
                function = str(
                    data.get(GraphAttributes.FUNCTION, "None")).strip()
                node_storage[n_type].setdefault(
                    "contracts", []).append(contract)
                node_storage[n_type].setdefault("functions", []).append(
                    f"{contract}_{function}"
                )

            # 2. Build Edge Lists
            edge_lists = defaultdict(
                lambda: {"src": [], "dst": [], "texts": []})

            for u, v, data in nx_graph.edges(data=True):
                if u not in global_to_local or v not in global_to_local:
                    print(
                        f"Warning: Edge ({u}->{v}) has missing nodes, skipping.")
                    continue

                src_type, src_idx = global_to_local[u]
                dst_type, dst_idx = global_to_local[v]
                e_type = str(
                    data.get(GraphAttributes.EDGE_TYPE, "connected_to")
                ).lower()

                rel = (src_type, e_type, dst_type)
                edge_lists[rel]["src"].append(src_idx)
                edge_lists[rel]["dst"].append(dst_idx)
                edge_lists[rel]["texts"].append(
                    f"{data.get('label', '')} {e_type} {data.get(GraphAttributes.SUB_EDGE_TYPE, '')}"
                )

            # 3. Create DGL Graph
            graph_data = {}
            for rel, data in edge_lists.items():
                graph_data[rel] = (torch.tensor(data["src"]),
                                   torch.tensor(data["dst"]))

            # Handle empty graph_data (no edges)
            if not graph_data:
                print(
                    "Warning: No edges in graph, adding dummy relation for DGL compatibility"
                )
                # Save graph to .dot for debug
                debug_dot_path = f"debug_no_edges_{len(nx_graph.nodes())}_nodes.dot"
                try:
                    nx.drawing.nx_pydot.write_dot(nx_graph, debug_dot_path)
                    print(f"Saved debug graph to {debug_dot_path}")
                except Exception as e:
                    print(f"Failed to save debug .dot: {e}")
                fallback_ntype = next(
                    iter(node_storage.keys()), self.cfg_node_label)
                graph_data = {
                    (fallback_ntype, "dummy", fallback_ntype): (
                        torch.tensor([], dtype=torch.long),
                        torch.tensor([], dtype=torch.long),
                    )
                }

            # Create heterograph with explicit node counts to include isolated nodes
            num_nodes_dict = {
                n_type: len(store["ids"]) for n_type, store in node_storage.items()
            }
            g = dgl.heterograph(graph_data, num_nodes_dict=num_nodes_dict)

            # 4. Generate & Assign Features
            for n_type, store in node_storage.items():
                # BERT
                bert_emb = self._get_bert_embeddings(store["texts"])
                # Struct (Normalize & Pad)
                struct_tensor = torch.tensor(
                    store["structs"], dtype=torch.float32)
                # Normalize
                if struct_tensor.shape[0] > 1:
                    mean = struct_tensor.mean(dim=0, keepdim=True)
                    std = struct_tensor.std(dim=0, keepdim=True) + 1e-6
                    struct_tensor = (struct_tensor - mean) / std

                # Concatenate
                g.nodes[n_type].data["feat"] = torch.cat(
                    [bert_emb, struct_tensor], dim=1
                )

                # Add IDs for Stage 1/2 pooling
                # Map contract/function names to integer IDs
                unique_contracts = sorted(list(set(store["contracts"])))
                unique_functions = sorted(list(set(store["functions"])))
                # print(unique_contracts)
                # print(unique_functions)

                contract_to_id = {c: i for i, c in enumerate(unique_contracts)}
                function_to_id = {f: i for i, f in enumerate(unique_functions)}

                contract_ids = [contract_to_id[c] for c in store["contracts"]]
                function_ids = [function_to_id[f] for f in store["functions"]]

                g.nodes[n_type].data["contract_id"] = torch.tensor(
                    contract_ids, dtype=torch.long
                )
                g.nodes[n_type].data["function_id"] = torch.tensor(
                    function_ids, dtype=torch.long
                )
                # Store original node IDs for Stage 3 label mapping
                g.nodes[n_type].data["node_ids"] = torch.tensor(
                    [int(nid) if str(nid).isdigit() else hash(str(nid)) %
                        (2**31) for nid in store["ids"]],
                    dtype=torch.long
                )

            for rel, store in edge_lists.items():
                if not store["texts"]:
                    continue
                g.edges[rel].data["feat"] = self._get_bert_embeddings(
                    store["texts"])

            # print(g)
            # g.global_to_local = global_to_local
            return g
        except Exception as e:
            print(f"Error in nx_to_dgl: {e}")
            traceback.print_exc()
            return None

    def _get_checkpoint_path(self, project_name: str) -> Path:
        """Get checkpoint file path for a project."""
        # Sanitize project name for filename
        safe_name = self._sanitize_name(project_name)
        return self.checkpoint_dir / f"{safe_name}.pt"

    def _sanitize_name(self, name: str) -> str:
        """Sanitize project name for safe filename."""
        return "".join(
            c if c.isalnum() or c in ("-", "_") else "_" for c in name
        )

    def _save_checkpoint(
        self, project_name: str, dgl_s1, dgl_s2, dgl_s3, s1_lbl, s2_lbl, s3_lbl
    ):
        """Save processed graphs and labels for a project."""
        print(f"  Saving checkpoint for project {project_name}...")
        checkpoint_path = self._get_checkpoint_path(project_name)
        checkpoint_data = {
            "project_name": project_name,
            "stage1_graph": dgl_s1,
            "stage2_graph": dgl_s2,
            "stage3_graph": dgl_s3,
            "stage1_labels": s1_lbl,
            "stage2_labels": s2_lbl,
            "stage3_labels": s3_lbl,
        }
        torch.save(checkpoint_data, checkpoint_path)
        print(f"  Saved checkpoint: {checkpoint_path}")

    def _checkpoint_is_compatible(self, checkpoint_data: Dict[str, Any]) -> bool:
        """Validate checkpoint schema matches current expected stage-keyed labels.

        Expected:
        - stageX_graph: dict[subkey -> DGLGraph]
        - stage1_labels/stage2_labels: dict[subkey -> int]
        - stage3_labels: dict[subkey -> {cfg_node: list, ast_node: list}]
        """
        try:
            for stage in ("stage1", "stage2", "stage3"):
                g_key = f"{stage}_graph"
                l_key = f"{stage}_labels"
                if g_key not in checkpoint_data or l_key not in checkpoint_data:
                    return False

                graphs = checkpoint_data[g_key]
                labels = checkpoint_data[l_key]
                if not isinstance(graphs, dict) or not isinstance(labels, dict):
                    return False

                graph_keys = set(graphs.keys())
                label_keys = set(labels.keys())
                # labels must cover graphs
                if graph_keys and not graph_keys.issubset(label_keys):
                    return False

                if stage == "stage3":
                    # stage3 label values must be dict with cfg_node/ast_node
                    for k in list(graph_keys)[:5]:
                        v = labels.get(k)
                        if not isinstance(v, dict):
                            return False
                        if "cfg_node" not in v or "ast_node" not in v:
                            return False
            return True
        except Exception:
            return False

    def _load_checkpoint(self, project_name: str):
        """Load processed graphs and labels for a project."""
        checkpoint_path = self._get_checkpoint_path(project_name)
        if checkpoint_path.exists():
            try:
                checkpoint_data = torch.load(
                    checkpoint_path, weights_only=False)
                return checkpoint_data
            except Exception as e:
                print(
                    f"  Warning: Failed to load checkpoint {checkpoint_path}: {e}")
                return None
        return None

    def _get_processed_projects(self) -> Set[str]:
        """Get set of already processed project names."""
        processed = set()
        if self.checkpoint_dir.exists():
            for checkpoint_file in self.checkpoint_dir.glob("*.pt"):
                processed.add(
                    str(checkpoint_file.name).replace(".pt", "").strip())
        return processed

    def _process_single_project(self, p_name, nx_g, vuln_data, *, return_results: bool = True):
        """Process a single project.

        Note: This function always saves a checkpoint on success.
        If return_results=False, it avoids returning large graph objects and returns only project_name.
        """
        print("=" * 40, "\n")
        print(f"Processing project: {p_name}")
        print("=" * 40, "\n")
        if not vuln_data:
            print(
                f"  Warning: No vulnerability data for project {p_name}, skipping."
            )
            return None

        try:
            # region main
            # 1. Build Metadata
            hierarchy = self._build_hierarchy_map(nx_g)

            # 2. Create Labels
            s3_lbl, s2_lbl, s1_lbl, original_vuln_count, vuln_map = (
                self._create_stage_labels(nx_g, hierarchy, vuln_data)
            )

            # 3. Pruning Cascade
            s1_g = self._prune_stage1(nx_g, s1_lbl, hierarchy, vuln_map)
            s2_g = self._prune_stage2(s1_g, s2_lbl, hierarchy, vuln_map)
            s3_g = self._prune_stage3(s2_g, vuln_data)

            s3_total_nodes = sum(
                g["graph"].number_of_nodes() for g in s3_g.values()
            )
            s3_total_edges = sum(
                g["graph"].number_of_edges() for g in s3_g.values()
            )

            print(
                f"    Stage3 Nodes={s3_total_nodes} Edges={s3_total_edges}")
            print("=" * 40)

            if s3_total_nodes == 0 or s3_total_edges == 0:
                print(
                    f"  Warning: Stage 3 graph is empty for project {p_name}, skipping."
                )
                # save empty txt for debug
                os.makedirs("./debug", exist_ok=True)
                with open(f"./debug/{p_name}_empty_stage3.txt", "w") as f:
                    f.write(
                        f"Project {p_name} has empty Stage 3 graph after pruning.\n"
                    )
                return None

            # 5. DGL Conversion (Distinct Graphs per Stage!)
            dgl_s1 = {k: self.nx_to_dgl(v["graph"])
                      for k, v in tqdm(s1_g.items(), desc="Conv Stage1")}
            dgl_s2 = {k: self.nx_to_dgl(v["graph"])
                      for k, v in tqdm(s2_g.items(), desc="Conv Stage2")}
            dgl_s3 = {k: self.nx_to_dgl(v["graph"])
                      for k, v in tqdm(s3_g.items(), desc="Conv Stage3")}

            # Move all tensors to CPU to avoid multiprocessing issues with CUDA
            def move_to_cpu(graph_dict):
                for g in graph_dict.values():
                    for ntype in g.ntypes:
                        for k, v in g.nodes[ntype].data.items():
                            if isinstance(v, torch.Tensor):
                                g.nodes[ntype].data[k] = v.cpu()
                    for etype in g.etypes:
                        for k, v in g.edges[etype].data.items():
                            if isinstance(v, torch.Tensor):
                                g.edges[etype].data[k] = v.cpu()

            move_to_cpu(dgl_s1)
            move_to_cpu(dgl_s2)
            move_to_cpu(dgl_s3)

            # Count vuln labels in stage3
            stage3_vuln_labels = 0
            for key, data in s3_g.items():
                for y in data["label"]["ast_node"]:
                    if sum(y) > 0:
                        stage3_vuln_labels += 1
                        break
                for x in data["label"]["cfg_node"]:
                    if sum(x) > 0:
                        stage3_vuln_labels += 1
                        break
            print(
                f"Vuln labels {stage3_vuln_labels}/{original_vuln_count} intact")
            # Build stage-keyed labels aligned with the pruned graph keys
            # - Stage 1: key is "{file}_{contract}"
            # - Stage 2: key is "{file}_{contract}_{function}"
            # - Stage 3: key is Stage 2 key (vulnerable functions only)
            s1_labels_by_key = {k: v.get("label", 0)
                                for k, v in s1_g.items()}
            s2_labels_by_key = {k: v.get("label", 0)
                                for k, v in s2_g.items()}
            s3_labels_by_key = {k: v.get("label", {})
                                for k, v in s3_g.items()}

            # Fail-fast contract validation (prevents silent schema drift)
            self._validate_project_stage_outputs(
                p_name, 1, dgl_s1, s1_labels_by_key)
            self._validate_project_stage_outputs(
                p_name, 2, dgl_s2, s2_labels_by_key)
            self._validate_project_stage_outputs(
                p_name, 3, dgl_s3, s3_labels_by_key)

            # 6. Save checkpoint (MUST match current schema)
            self._save_checkpoint(
                p_name, dgl_s1, dgl_s2, dgl_s3, s1_labels_by_key, s2_labels_by_key, s3_labels_by_key
            )

            if return_results:
                # 7. Prepare results
                project_results = {
                    "stage1": {"graphs": dgl_s1, "labels": s1_labels_by_key},
                    "stage2": {"graphs": dgl_s2, "labels": s2_labels_by_key},
                    "stage3": {"graphs": dgl_s3, "labels": s3_labels_by_key},
                }

                print(
                    f"Processed {p_name}: S1Nodes={sum(g.num_nodes() for g in dgl_s1.values())}, S3Nodes={sum(g.num_nodes() for g in dgl_s3.values())}"
                )

            # Free memory: delete large intermediate objects
            del hierarchy, s3_lbl, s2_lbl, s1_lbl, original_vuln_count, vuln_map
            del s1_g, s2_g, s3_g, dgl_s1, dgl_s2, dgl_s3
            del s1_labels_by_key, s2_labels_by_key, s3_labels_by_key
            del nx_g  # delete the input graph
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            return (p_name, project_results) if return_results else p_name

        except Exception as e:
            print(f"\n  Error processing project {p_name}: {e}")
            traceback.print_exc()
            print(f"  Skipping {p_name} and continuing...\n")
            return None

    def process_graphs(self, nx_graphs_list, vuln_json_list, resume=True):
        """Main Pipeline execution with checkpointing.

        Args:
            nx_graphs_list: List of (project_name, networkx_graph) tuples
            vuln_json_list: List of (project_name, vulnerability_data) tuples
            resume: If True, skip already processed projects and load from checkpoints
        """
        print(f"Processing {len(nx_graphs_list)} projects...")

        # Build Lookup
        vuln_lookup = {item[0]: item[1] for item in vuln_json_list}
        # Note: keep nx_graphs_list as the source of truth; no need to build an extra lookup here.

        # Check for already processed projects
        processed_projects = self._get_processed_projects() if resume else set()
        if processed_projects:
            print(
                f"Found {len(processed_projects)} already processed projects.")

        # Separate projects into to-process and already-done
        projects_to_process = []
        projects_to_load = []

        for item in nx_graphs_list:
            p_name, nx_g = item
            if resume and self._sanitize_name(p_name) in processed_projects:
                projects_to_load.append(p_name)
            else:
                projects_to_process.append(item)

        print(
            f"Found {len(projects_to_load)} completed projects (will load after processing), processing {len(projects_to_process)} new projects."
        )

        # Verify checkpointed projects exist (don't load yet to prevent OOM)
        verified_checkpoints = []
        for p_name in tqdm(projects_to_load, desc="Verifying checkpoints"):
            checkpoint_path = self._get_checkpoint_path(p_name)
            if checkpoint_path.exists():
                verified_checkpoints.append(p_name)
            else:
                # If checkpoint missing, add back to processing queue
                print(f"  Checkpoint missing for {p_name}, will reprocess.")
                # Find original graph
                for item in nx_graphs_list:
                    if item[0] == p_name:
                        projects_to_process.append(item)
                        break

        print(f"Verified {len(verified_checkpoints)} valid checkpoints.")
        processed_now = []
        # Process remaining projects in parallel (to speed up)
        if projects_to_process:
            # Adjust based on GPU/CPU; keep small to reduce CUDA IPC pressure
            num_processes = min(len(projects_to_process), 2)
            print(f"Using {num_processes} processes for parallel processing.")
            mp_ctx = multiprocessing.get_context(
                "spawn") if torch.cuda.is_available() else multiprocessing.get_context()
            with concurrent.futures.ProcessPoolExecutor(max_workers=num_processes, mp_context=mp_ctx) as executor:
                worker_func = partial(
                    process_project_worker,
                    vuln_lookup=vuln_lookup,
                    device=self.device,
                    batch_size=self.batch_size,
                    checkpoint_dir=self.checkpoint_dir,
                    embedding_model_conf=self.embedding_model_conf,
                )
                futures = [executor.submit(worker_func, item)
                           for item in projects_to_process]
                results_list = [f.result()
                                for f in concurrent.futures.as_completed(futures)]

            # Collect results (project names)
            processed_now = [res for res in results_list if res]

            projects_to_process = []  # Already processed in parallel

        # No temp partial files: rely on checkpoints as the durable intermediate.
        all_available = list(dict.fromkeys(list(verified_checkpoints) + list(processed_now)))
        return all_available
