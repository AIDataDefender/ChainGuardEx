from torch.utils.data import Dataset
from datetime import datetime
from tqdm import tqdm
import torch
import torch.nn as nn
import numpy as np
import random
import json
import sys
import os
import traceback
import dgl
import argparse


# Add the workspace root to Python path
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

try:
    from experiments.the_utils.logger import setup_logger
    from experiments.cpg_processor import CPG_Processor
    from experiments.the_utils import graph_utils as _graph_utils
    from Data.DAppSCAN.f6_DAppSCAN_fetch_data_adapter import f6_fetch_DAppSCAN_data
except Exception:
    from the_utils.logger import setup_logger
    from cpg_processor import CPG_Processor
    from the_utils import graph_utils as _graph_utils
    from Data.DAppSCAN.f6_DAppSCAN_fetch_data_adapter import f6_fetch_DAppSCAN_data


# Logging Setup
log_folder = os.getenv("LOG_FOLDER", "Logs")
os.makedirs(log_folder, exist_ok=True)
logger = setup_logger(f"{log_folder}/Dataset.log")

os.environ["DGLBACKEND"] = "pytorch"
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def normalize_graph_keys(data):
    """
    Recursively converts dictionary keys to strings for DGL compatibility.
    Handles Tuples (canonical edge types) and Enums.
    """
    if isinstance(data, dict):
        new_data = {}
        for k, v in data.items():
            if isinstance(k, tuple):
                k_str = tuple(
                    elem.value if hasattr(elem, "value") else str(elem) for elem in k
                )
            elif hasattr(k, "value"):
                k_str = str(k.value)
            else:
                k_str = str(k) if not isinstance(k, str) else k
            new_data[k_str] = normalize_graph_keys(v)
        return new_data
    elif isinstance(data, list):
        return [normalize_graph_keys(item) for item in data]
    elif isinstance(data, tuple):
        return tuple(normalize_graph_keys(item) for item in data)
    else:
        return data


class GraphEmbedder(nn.Module):
    def __init__(self, embedding_dim=768, num_heads=4, seed=42, node_in_dims=None):
        super(GraphEmbedder, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.seed = seed
        self._warned_unknown_dim = False

        # Optional per-node-type projection so we can consume full node feature vectors
        # (e.g., 806/834) while still outputting a stable embedding_dim (768).
        self.node_proj = nn.ModuleDict()
        if isinstance(node_in_dims, dict):
            for ntype, in_dim in node_in_dims.items():
                try:
                    in_dim_int = int(in_dim)
                except Exception:
                    continue
                if in_dim_int > 0 and in_dim_int != self.embedding_dim:
                    self.node_proj[ntype] = nn.Linear(in_dim_int, self.embedding_dim)

    def _to_embed_dim(self, feat: torch.Tensor, ntype: str) -> torch.Tensor:
        """Project/pad/slice feat to [N, embedding_dim] without losing info when possible."""
        if not isinstance(feat, torch.Tensor) or feat.dim() != 2:
            raise TypeError(f"Expected 2D Tensor feat for {ntype}, got {type(feat)}")

        # Preferred path: learned projection using full input width.
        if ntype in self.node_proj:
            return self.node_proj[ntype](feat)

        # Fallback path (should be rare): pad/slice to match embedding_dim.
        in_dim = feat.shape[1]
        if in_dim == self.embedding_dim:
            return feat

        if not self._warned_unknown_dim:
            logger.warning(
                f"GraphEmbedder: missing projection for ntype={ntype} in_dim={in_dim}; falling back to pad/slice."
            )
            self._warned_unknown_dim = True

        if in_dim > self.embedding_dim:
            return feat[:, : self.embedding_dim]
        pad = torch.zeros(
            (feat.shape[0], self.embedding_dim - in_dim),
            device=feat.device,
            dtype=feat.dtype,
        )
        return torch.cat([feat, pad], dim=1)

    def forward(self, g):
        # Pre-compute graph embedding by multi-head attention pooling on all node features
        all_node_feats = []
        moved_to = None
        if isinstance(g, dict):
            # g is dict of {contract: dgl_graph}
            for _, graph in g.items():
                if hasattr(graph, "ntypes"):
                    for ntype in graph.ntypes:
                        feat = None
                        try:
                            feat = graph.nodes[ntype].data.get("feat")
                        except Exception:
                            feat = None
                        if isinstance(feat, torch.Tensor) and feat.numel() > 0:
                            # Ensure projection weights live on the same device as features.
                            if moved_to is None and any(True for _ in self.parameters()):
                                param_dev = next(self.parameters()).device
                                if param_dev != feat.device:
                                    self.to(feat.device)
                            moved_to = feat.device
                            all_node_feats.append(self._to_embed_dim(feat, str(ntype)))
                else:
                    # Homogeneous fallback
                    feat = getattr(graph, "ndata", {}).get("feat") if hasattr(graph, "ndata") else None
                    if isinstance(feat, torch.Tensor) and feat.numel() > 0:
                        if moved_to is None and any(True for _ in self.parameters()):
                            param_dev = next(self.parameters()).device
                            if param_dev != feat.device:
                                self.to(feat.device)
                        moved_to = feat.device
                        all_node_feats.append(self._to_embed_dim(feat, "_homogeneous"))
        else:
            # g is single DGL graph
            if hasattr(g, "ntypes"):
                for ntype in g.ntypes:
                    feat = None
                    try:
                        feat = g.nodes[ntype].data.get("feat")
                    except Exception:
                        feat = None
                    if isinstance(feat, torch.Tensor) and feat.numel() > 0:
                        if moved_to is None and any(True for _ in self.parameters()):
                            param_dev = next(self.parameters()).device
                            if param_dev != feat.device:
                                self.to(feat.device)
                        moved_to = feat.device
                        all_node_feats.append(self._to_embed_dim(feat, str(ntype)))
            else:
                # Homogeneous fallback
                feat = getattr(g, "ndata", {}).get("feat") if hasattr(g, "ndata") else None
                if isinstance(feat, torch.Tensor) and feat.numel() > 0:
                    if moved_to is None and any(True for _ in self.parameters()):
                        param_dev = next(self.parameters()).device
                        if param_dev != feat.device:
                            self.to(feat.device)
                    moved_to = feat.device
                    all_node_feats.append(self._to_embed_dim(feat, "_homogeneous"))
        if all_node_feats:
            # [total_nodes, self.embedding_dim]
            combined_feats = torch.cat(all_node_feats, dim=0)
            torch.manual_seed(self.seed)  # For reproducible queries
            queries = [torch.randn(self.embedding_dim, dtype=torch.float32,
                                   device=combined_feats.device) for _ in range(self.num_heads)]
            pooled = []
            for query in queries:
                attn_logits = combined_feats @ query  # [total_nodes]
                attn = torch.softmax(attn_logits, dim=0)  # [total_nodes]
                # [self.embedding_dim]
                pooled.append((attn.unsqueeze(1) * combined_feats).sum(dim=0))
            # [self.num_heads * self.embedding_dim]
            graph_embedding = torch.cat(pooled, dim=0)
        else:
            # Fallback if no features
            graph_embedding = torch.zeros(
                self.num_heads * self.embedding_dim, dtype=torch.float32)
        return graph_embedding


class CustomDataset(Dataset):
    """
    3-Stage Cascaded Dataset.
    Stage 1: Contract-Level (Binary)
    Stage 2: Function-Level (Binary)
    Stage 3: Block-Level (Multi-label Node Classification)
    """

    def __init__(
        self,
        source="DAppSCAN",
        force_reload=False,
        rand_seed=42,
        stage=3,
        is_test=False,
    ):
        logger.info(f"Initializing CustomDataset for Stage {stage}...")
        self.stage = stage
        if stage not in [1, 2, 3]:
            raise ValueError(f"Invalid stage {stage}. Must be 1, 2, or 3.")

        self._set_rand_seed(rand_seed)
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

        self.dataset_graph = {}
        self.dataset_label = {}
        self.is_test = is_test

        # Initialize Processor components only if needed (Force Reload or Missing Data)
        # We check simple existence first to avoid loading heavy models unnecessarily
        base_filename = "DAppSCAN_dataset.pt"
        self.save_load_dir = "./save_data"
        os.makedirs(self.save_load_dir, exist_ok=True)

        # Check if we need to process
        needs_processing = force_reload
        if not needs_processing:
            # Check if specific stage files exist (graph + label)
            test_suffix = "test" if self.is_test else ""
            g_path = os.path.join(
                self.save_load_dir,
                base_filename.replace(
                    ".pt", f"_stage{stage}{test_suffix}_graph.pt"),
            )
            l_path = os.path.join(
                self.save_load_dir,
                base_filename.replace(
                    ".pt", f"_stage{stage}{test_suffix}_label.pt"),
            )
            if not (os.path.exists(g_path) and os.path.exists(l_path)):
                needs_processing = True
                logger.info(
                    f"Stage {stage} data not found. Triggering processing.")
        self.tokenizer = None
        self.embedding_model = None
        if needs_processing:
            from transformers import AutoModel, AutoTokenizer

            logger.info("Loading CodeBERT for data processing...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                "microsoft/codebert-base", use_fast=True
            )
            self.embedding_model = AutoModel.from_pretrained(
                "microsoft/codebert-base")
        else:
            logger.info("Loading mode: Skipping CodeBERT initialization.")

        self.CPG_Proccessor = CPG_Processor(
            tokenizer=self.tokenizer,
            model=self.embedding_model,
            device=self.device,
            batch_size=16,
            checkpoint_dir="./checkpoints/processed_graphs",
        )
        self.embedding_dims = self.CPG_Proccessor.embedding_dims

        # Stage1/2 embedder: use full node feat widths via projection to 768,
        # keeping downstream embedding size stable at 3072.
        node_in_dims = {}
        for k in ("cfg_node", "ast_node"):
            if k in self.embedding_dims:
                node_in_dims[k] = int(self.embedding_dims[k])
        self.graph_embedder = GraphEmbedder(
            embedding_dim=768, num_heads=4, node_in_dims=node_in_dims
        ).to(self.device)
        print(f"  Embedding Dims: {json.dumps(self.embedding_dims, indent=2)}")
        if source == "DAppSCAN":
            self._fetch_and_process_DAppSCAN_data(
                force_reload=needs_processing)
            self.postprocessing()

    def set_active_stage(self, stage):
        self.stage = stage
        if stage not in [1, 2, 3]:
            raise ValueError(f"Invalid stage {stage}. Must be 1, 2, or 3.")

    def _set_rand_seed(self, rand_seed):
        random.seed(rand_seed)
        np.random.seed(rand_seed)
        torch.manual_seed(rand_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(rand_seed)

    def _fetch_and_process_DAppSCAN_data(self, force_reload=False):
        """Orchestrates loading from disk or fetching & processing data."""
        base_filename = "DAppSCAN_dataset.pt"
        saved_file_path = os.path.join(self.save_load_dir, base_filename)

        try:
            # 1. Try Loading Existing Data (If not forced)
            if not force_reload:
                if self._load_data(saved_file_path):
                    return

            # 2. Process New Data (Generates ALL stages)
            logger.info("Fetching and processing DAppSCAN data...")
            res = f6_fetch_DAppSCAN_data(
                root=r"Data/DAppSCAN/ProcessedData/success", is_test=self.is_test
            )

            if not res or not res[0] or not res[1]:
                logger.error("Failed to fetch DAppSCAN data.")
                sys.exit(1)

            cpg_list, vuln_json_list = res

            # Run 3-Stage Processing
            # This returns a dict with 'stage1', 'stage2', 'stage3' keys
            logger.info("Running CPG Processor (generating all 3 stages)...")
            self.cascade_results = self.CPG_Proccessor.process_graphs(
                cpg_list, vuln_json_list
            )

            # 3. Save All Stages
            self._save_data(saved_file_path)

            # 4. Populate in-memory for the active stage (so postprocessing works)
            stage_key = f"stage{self.stage}"
            stage_data = self.cascade_results.get(stage_key, {})
            # Store as dicts keyed by project name to avoid relying on list ordering.
            self.dataset_graph[self.stage] = dict(stage_data.get("graphs", []))
            self.dataset_label[self.stage] = dict(stage_data.get("labels", []))

        except Exception as e:
            logger.error(f"Error in data processing: {e}")
            traceback.print_exc()

    def _save_data(self, base_path):
        """Saves distinct files for each stage."""
        try:
            logger.info("Saving 3-stage cascade data to disk...")
            for s in [1, 2, 3]:
                key = f"stage{s}"
                data = self.cascade_results.get(key, {})

                # Paths
                test_suffix = "test" if self.is_test else ""
                g_path = base_path.replace(
                    ".pt", f"_stage{s}{test_suffix}_graph.pt"
                )
                l_path = base_path.replace(
                    ".pt", f"_stage{s}{test_suffix}_label.pt"
                )

                if "graphs" in data:
                    flattened_graphs = []
                    flattened_labels = []
                    # Store as triples to avoid unsafe parsing/splitting on underscores.
                    # flattened_graphs: List[(project_name, subkey, graph)]
                    # flattened_labels: List[(project_name, subkey, label)]
                    for p_name, graph_dict in data.get("graphs", []):
                        if not isinstance(graph_dict, dict):
                            logger.warning(
                                f"Stage {s} graphs for {p_name} is not a dict; skipping"
                            )
                            continue
                        for subkey, g in graph_dict.items():
                            flattened_graphs.append((p_name, subkey, g))
                    for p_name, label_dict in data.get("labels", []):
                        if not isinstance(label_dict, dict):
                            logger.warning(
                                f"Stage {s} labels for {p_name} is not a dict; skipping"
                            )
                            continue
                        for subkey, lbl in label_dict.items():
                            flattened_labels.append((p_name, subkey, lbl))

                    torch.save(flattened_graphs, g_path)
                    torch.save(flattened_labels, l_path)
                    logger.info(
                        f"  Saved Stage {s}: {len(flattened_graphs)} items")
                else:
                    logger.warning(f"  No data found for Stage {s} to save.")

        except Exception as e:
            logger.error(f"Failed to save data: {e}")

    def _load_data(self, base_path):
        """Loads the current stage file."""
        s = self.stage
        print(f"Attempting to load Stage {s} data from disk...")
        test_suffix = "test" if self.is_test else ""
        g_path = base_path.replace(".pt", f"_stage{s}{test_suffix}_graph.pt")
        l_path = base_path.replace(".pt", f"_stage{s}{test_suffix}_label.pt")

        if os.path.exists(g_path) and os.path.exists(l_path):
            logger.info(f"Loading Stage {s} from disk...")
            flattened_graphs = torch.load(
                g_path, map_location="cpu", weights_only=False
            )
            flattened_labels = torch.load(
                l_path, map_location="cpu", weights_only=False
            )

            # Deflatten: Group by project (supports both new triple format and legacy combined-key format)
            graphs_by_project = {}
            for item in flattened_graphs:
                if isinstance(item, (list, tuple)) and len(item) == 3:
                    p_name, subkey, g = item
                elif isinstance(item, (list, tuple)) and len(item) == 2:
                    # HARD: reject legacy combined-key format (unsafe to parse).
                    logger.warning(
                        f"Stage {s} graph file is legacy 2-tuple format; refusing to load. Please reprocess with --force_reload."
                    )
                    return False
                else:
                    logger.warning(
                        f"Unexpected graph item format: {type(item)}")
                    continue
                graphs_by_project.setdefault(p_name, {})[subkey] = g

            labels_by_project = {}
            for item in flattened_labels:
                if isinstance(item, (list, tuple)) and len(item) == 3:
                    p_name, subkey, lbl = item
                elif isinstance(item, (list, tuple)) and len(item) == 2:
                    logger.warning(
                        f"Stage {s} label file is legacy 2-tuple format; refusing to load. Please reprocess with --force_reload."
                    )
                    return False
                else:
                    logger.warning(
                        f"Unexpected label item format: {type(item)}")
                    continue
                labels_by_project.setdefault(p_name, {})[subkey] = lbl

            # Store as dicts keyed by project to guarantee correct alignment in postprocessing.
            self.dataset_graph[s] = graphs_by_project
            self.dataset_label[s] = labels_by_project

            # Validate schema: labels must cover graphs for this stage.
            # If not, treat saved files as incompatible (likely produced by older pipeline).
            missing_total = 0
            total_graph_items = 0
            for p_name, g_dict in graphs_by_project.items():
                if not isinstance(g_dict, dict):
                    continue
                graph_keys = set(g_dict.keys())
                total_graph_items += len(graph_keys)
                label_keys = set(labels_by_project.get(p_name, {}).keys())
                missing_total += len(graph_keys - label_keys)

            if total_graph_items > 0 and missing_total > 0:
                logger.warning(
                    f"Stage {s} saved data appears incompatible: missing {missing_total}/{total_graph_items} labels. Will reprocess."
                )
                # Clear any partially loaded data for this stage.
                self.dataset_graph.pop(s, None)
                self.dataset_label.pop(s, None)
                return False

            logger.info(
                f"  Loaded {len(graphs_by_project)} projects for Stage {s}."
            )
            return True
        else:
            logger.warning(f"Stage {s} files not found at {g_path}")
            return False

    def postprocessing(self):
        dataset_graph_temp = []
        dataset_label_temp = []

        if self.dataset_graph is None or self.dataset_label is None:
            logger.error(
                "Dataset graph or label is None during postprocessing.")
            return

        if self.stage not in self.dataset_graph or self.stage not in self.dataset_label:
            logger.error(
                f"Stage {self.stage} data not loaded/available for postprocessing.")
            return

        raw_graphs = self.dataset_graph[self.stage]
        raw_labels = self.dataset_label[self.stage]

        # Accept either dicts keyed by project or list[(project, dict)] (legacy/in-memory).
        graphs_by_project = dict(raw_graphs) if isinstance(raw_graphs, list) else raw_graphs
        labels_by_project = dict(raw_labels) if isinstance(raw_labels, list) else raw_labels

        if not isinstance(graphs_by_project, dict) or not isinstance(labels_by_project, dict):
            logger.error(
                f"Unexpected stage containers: graphs={type(graphs_by_project)} labels={type(labels_by_project)}"
            )
            return

        for p_name_g, graphs_data in graphs_by_project.items():
            labels_data = labels_by_project.get(p_name_g, {})

            if not isinstance(graphs_data, dict) or not isinstance(labels_data, dict):
                logger.warning(
                    f"Unexpected per-project data types for {p_name_g}: graphs={type(graphs_data)} labels={type(labels_data)}"
                )
                continue

            # Align graph+label per subkey (so __getitem__ is always consistent)
            for subkey, graph_data in graphs_data.items():
                full_key = f"{p_name_g}@{subkey}"

                lbl = labels_data.get(subkey)
                if lbl is None:
                    raise ValueError(
                        f"Missing label for {full_key}. This indicates stage graph/label keys are inconsistent."
                    )

                g = normalize_graph_keys(graph_data)
                if self.stage in [1, 2]:
                    # HARD: stage1/2 labels must be binary.
                    if isinstance(lbl, bool):
                        lbl = int(lbl)
                    if not isinstance(lbl, int) or lbl not in (0, 1):
                        raise ValueError(
                            f"Stage {self.stage} label must be int 0/1 for {full_key}, got {lbl} ({type(lbl)})"
                        )
                    g = g.to(self.device)
                    g = self.graph_embedder(g)
                    if not isinstance(g, torch.Tensor) or g.dim() != 1:
                        raise TypeError(
                            f"Stage {self.stage} embedder output must be 1D tensor for {full_key}, got {type(g)}"
                        )
                    expected_dim = int(self.graph_embedder.num_heads * self.graph_embedder.embedding_dim)
                    if g.shape[0] != expected_dim:
                        raise ValueError(
                            f"Stage {self.stage} embedding dim mismatch for {full_key}: got {g.shape[0]} expected {expected_dim}"
                        )
                else:
                    # Stage 3: fail-fast contract check: label rows must match node counts.
                    if not hasattr(g, "ntypes"):
                        raise TypeError(f"Stage 3 expects a DGL heterograph, got {type(g)} for {full_key}")

                    if not isinstance(lbl, dict) or ("cfg_node" not in lbl or "ast_node" not in lbl):
                        raise TypeError(
                            f"Stage 3 label must be dict with cfg_node/ast_node for {full_key}, got {type(lbl)}"
                        )

                    expected_dim = len(getattr(_graph_utils, "OWASP_VULN", []))
                    if expected_dim <= 0:
                        raise RuntimeError("OWASP_VULN is not defined or empty")

                    def _validate_label_rows(rows, node_type: str):
                        if rows is None:
                            return
                        if not isinstance(rows, list):
                            raise TypeError(
                                f"Stage 3 {node_type} labels must be a list for {full_key}, got {type(rows)}"
                            )
                        for i, row in enumerate(rows[:5]):
                            if not isinstance(row, (list, tuple)):
                                raise TypeError(
                                    f"Stage 3 {node_type}[{i}] must be list/tuple for {full_key}, got {type(row)}"
                                )
                            if len(row) != expected_dim:
                                raise ValueError(
                                    f"Stage 3 {node_type}[{i}] dim mismatch for {full_key}: got {len(row)} expected {expected_dim}"
                                )
                            bad = [v for v in row if v not in (0, 1, 0.0, 1.0)]
                            if bad:
                                raise ValueError(
                                    f"Stage 3 {node_type}[{i}] has non-binary values for {full_key} (sample={bad[:5]})"
                                )

                    _validate_label_rows(lbl.get("cfg_node"), "cfg_node")
                    _validate_label_rows(lbl.get("ast_node"), "ast_node")

                    def _pick_ntype(graph, candidates):
                        for cand in candidates:
                            if cand in graph.ntypes:
                                return cand
                        return None

                    cfg_ntype = _pick_ntype(g, ["cfg_node", "node_cfg"])
                    ast_ntype = _pick_ntype(g, ["ast_node", "node_ast"])
                    cfg_nodes = g.num_nodes(cfg_ntype) if cfg_ntype is not None else 0
                    ast_nodes = g.num_nodes(ast_ntype) if ast_ntype is not None else 0

                    cfg_rows = int(lbl.get("cfg_node", []) and len(lbl.get("cfg_node", [])) or 0)
                    ast_rows = int(lbl.get("ast_node", []) and len(lbl.get("ast_node", [])) or 0)
                    if cfg_rows != cfg_nodes:
                        raise ValueError(
                            f"Stage 3 contract mismatch for {full_key}: cfg labels rows {cfg_rows} != {cfg_ntype} nodes {cfg_nodes}"
                        )
                    if ast_rows != ast_nodes:
                        raise ValueError(
                            f"Stage 3 contract mismatch for {full_key}: ast labels rows {ast_rows} != {ast_ntype} nodes {ast_nodes}"
                        )

                dataset_graph_temp.append((full_key, g))

                if self.stage in [1, 2]:
                    dataset_label_temp.append(
                        (full_key, torch.tensor(lbl, dtype=torch.float32))
                    )
                else:
                    dataset_label_temp.append(
                        (
                            full_key,
                            {
                                "cfg_node": torch.tensor(
                                    lbl.get("cfg_node", []), dtype=torch.float32
                                ),
                                "ast_node": torch.tensor(
                                    lbl.get("ast_node", []), dtype=torch.float32
                                ),
                            },
                        )
                    )

        if len(dataset_graph_temp) != len(dataset_label_temp):
            logger.warning(
                f"Postprocessing mismatch: graphs={len(dataset_graph_temp)} labels={len(dataset_label_temp)}"
            )

        self.dataset_graph = dataset_graph_temp
        self.dataset_label = dataset_label_temp
        print("Postprocessing done!")

    def __len__(self):
        return len(self.dataset_graph)

    def __getitem__(self, idx):
        k, g = self.dataset_graph[idx]
        p_name = k.split("@")[0]
        if self.stage in [1, 2]:
            label = self.dataset_label[idx][1]
            if self.dataset_label[idx][0] != k:
                raise ValueError("Graph and label keys do not match.")
            return {"embeddings": g, "graph_labels": {k: label}, "project_name": p_name}
        else:
            label_key, label_dict = self.dataset_label[idx]
            if label_key != k:
                raise ValueError("Graph and label keys do not match.")
            return {
                "graph": g,
                "cfg_labels": label_dict["cfg_node"],
                "ast_labels": label_dict["ast_node"],
                "project_name": p_name,
            }


def custom_collate(batch, stage=3):
    """
    Robust collate function for 3-Stage Pipeline.
    """
    try:
        # 1. Clean Batch
        batch = [b for b in batch if b is not None]
        if not batch:
            return None

        # 2. Batch Graphs (All graphs have consistent schema from CPG_Processor)
        graphs = [b["graph"] for b in batch if "graph" in b]
        batched_graph = dgl.batch(graphs) if graphs else None

        # Preserve metadata for Stage 1/2 label matching
        # Store per-graph ID-to-name mappings
        id_to_name_maps = []
        for g in graphs:
            # Try to get the metadata from original graph
            contract_map = getattr(g, "contract_id_to_name", None)
            function_map = getattr(g, "function_id_to_name", None)
            id_to_name_maps.append(
                {"contract": contract_map, "function": function_map})
        batched_graph.id_to_name_maps = id_to_name_maps

        # 3. Collate Labels
        def collate_key(key):
            items = [b.get(key) for b in batch if b.get(key) is not None]

            # A. Tensor Mode (Stage 3 Node Labels) -> Concat
            if all(isinstance(x, torch.Tensor) for x in items):
                return torch.cat(items, dim=0) if items else None

            # B. Dict/Raw Mode (Stage 1/2 Labels) -> List
            return items

        # Stage-specific collation
        if stage == 3:
            labels = {
                "cfg_labels": collate_key("cfg_labels"),
                "ast_labels": collate_key("ast_labels"),
            }
        else:
            # Stage 1/2: Embeddings and labels
            embeddings = torch.stack([b["embeddings"] for b in batch])
            labels = {"graph_labels": [b.get("graph_labels") for b in batch]}

        return {
            "graph": batched_graph if stage == 3 else None,
            "embeddings": embeddings if stage != 3 else None,
            **labels,
            "project_names": [b["project_name"] for b in batch],
        }
    except Exception as e:
        logger.error(f"Error in custom_collate: {e}")
        return None


if __name__ == "__main__":
    # Command Line Interface for Data Processing
    parser = argparse.ArgumentParser(description="DAppSCAN Dataset Processor")
    parser.add_argument(
        "--force_reload",
        action="store_true",
        help="If set, fully re-processes raw data and regenerates graphs for all 3 stages.",
    )
    parser.add_argument(
        "--stage",
        type=int,
        default=3,
        choices=[1, 2, 3],
        help="Which stage to verify/load after processing (default: 3).",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="If set, processes only a small test subset of the data.",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("Running Dataset Manager")
    print(f"  Force Reload: {args.force_reload}")
    print(f"  Target Stage: {args.stage}")
    print("=" * 60)

    def _stage_paths(stage: int):
        base = os.path.join("./save_data", "DAppSCAN_dataset.pt")
        test_suffix = "test" if args.test else ""
        g_path = base.replace(".pt", f"_stage{stage}{test_suffix}_graph.pt")
        l_path = base.replace(".pt", f"_stage{stage}{test_suffix}_label.pt")
        return g_path, l_path

    def _load_stage_files(stage: int):
        g_path, l_path = _stage_paths(stage)
        if not (os.path.exists(g_path) and os.path.exists(l_path)):
            return None, None

        flattened_graphs = torch.load(g_path, map_location="cpu", weights_only=False)
        flattened_labels = torch.load(l_path, map_location="cpu", weights_only=False)

        graphs_by_project = {}
        for item in flattened_graphs:
            if isinstance(item, (list, tuple)) and len(item) == 3:
                p_name, subkey, g = item
            else:
                raise ValueError(
                    f"Unexpected graph item format in {g_path}: {type(item)} len={len(item) if isinstance(item,(list,tuple)) else 'n/a'}"
                )
            graphs_by_project.setdefault(p_name, {})[subkey] = g

        labels_by_project = {}
        for item in flattened_labels:
            if isinstance(item, (list, tuple)) and len(item) == 3:
                p_name, subkey, lbl = item
            else:
                raise ValueError(
                    f"Unexpected label item format in {l_path}: {type(item)} len={len(item) if isinstance(item,(list,tuple)) else 'n/a'}"
                )
            labels_by_project.setdefault(p_name, {})[subkey] = lbl

        return graphs_by_project, labels_by_project

    def _print_and_assert_raw_schema(stage: int):
        graphs_by_project, labels_by_project = _load_stage_files(stage)
        if graphs_by_project is None or labels_by_project is None:
            print(f"Stage {stage}: no saved files")
            return

        projects = list(graphs_by_project.keys())
        print(f"\nFinal Results Schema (from saved stage{stage} files):")
        print(f"  stage{stage}: graphs={len(projects)}, labels={len(labels_by_project)}")
        if not projects:
            return

        p0 = projects[0]
        gdict = graphs_by_project[p0]
        ldict = labels_by_project.get(p0, {})
        print(f"    Sample project: {p0}")
        print(f"    Sample graphs container type: {type(gdict)}")
        print(f"    Subgraphs: {len(gdict)}")
        first_subkey = next(iter(gdict.keys())) if gdict else None
        print(f"    First subkey: {first_subkey}")
        if first_subkey is not None:
            print(f"    First subgraph type: {type(gdict[first_subkey])}")

        print(f"    Sample labels container type: {type(ldict)}")
        print(f"    Label keys: {len(ldict)}")
        if first_subkey is None:
            return

        # Strict key match
        g_keys = set(gdict.keys())
        l_keys = set(ldict.keys())
        if g_keys != l_keys:
            missing = sorted(list(g_keys - l_keys))[:10]
            extra = sorted(list(l_keys - g_keys))[:10]
            raise AssertionError(
                f"stage{stage} raw schema mismatch for project {p0}: graphs={len(g_keys)} labels={len(l_keys)} missing={missing} extra={extra}"
            )

        v0 = ldict[first_subkey]
        print(f"    First label key: {first_subkey}")
        print(f"    First label type: {type(v0)}")
        if stage == 3:
            if not isinstance(v0, dict) or ("cfg_node" not in v0 or "ast_node" not in v0):
                raise AssertionError(
                    f"stage3 label for {first_subkey} must be dict with cfg_node/ast_node, got {type(v0)}"
                )

    def _validate_postprocessed(ds: CustomDataset, stage: int, max_checks: int = 25):
        n = len(ds)
        print(f"Stage {stage} postprocessed samples: {n}")
        if n == 0:
            raise AssertionError(f"Stage {stage} produced 0 samples after postprocessing")

        checks = min(n, max_checks)
        for i in range(checks):
            item = ds[i]
            if stage in [1, 2]:
                label_dict = item["graph_labels"]
                if len(label_dict) != 1:
                    raise AssertionError(
                        f"Stage {stage}: expected 1 label per item, got {len(label_dict)}"
                    )
                k = next(iter(label_dict.keys()))
                if k.split("@")[0] != item["project_name"]:
                    raise AssertionError(
                        f"Stage {stage}: project mismatch for key {k}"
                    )
            else:
                if dgl is None:
                    raise RuntimeError("DGL is required to validate stage 3 graphs")
                g = item["graph"]
                cfg = item["cfg_labels"]
                ast = item["ast_labels"]
                def _pick_ntype(graph, candidates):
                    for cand in candidates:
                        if cand in graph.ntypes:
                            return cand
                    return None

                cfg_ntype = _pick_ntype(g, ["cfg_node", "node_cfg"])
                ast_ntype = _pick_ntype(g, ["ast_node", "node_ast"])
                cfg_nodes = g.num_nodes(cfg_ntype) if cfg_ntype is not None else 0
                ast_nodes = g.num_nodes(ast_ntype) if ast_ntype is not None else 0
                if cfg.shape[0] != cfg_nodes:
                    raise AssertionError(
                        f"Stage 3: cfg_labels rows {cfg.shape[0]} != {cfg_ntype} {cfg_nodes}"
                    )
                if ast.shape[0] != ast_nodes:
                    raise AssertionError(
                        f"Stage 3: ast_labels rows {ast.shape[0]} != {ast_ntype} {ast_nodes}"
                    )

        print(f"  Validation OK on {checks} samples")

    # One processing pass generates and saves all 3 stages.
    _ = CustomDataset(stage=3, force_reload=args.force_reload, is_test=args.test)

    # Print and assert raw saved schema for all 3 stages.
    for s in [1, 2, 3]:
        _print_and_assert_raw_schema(s)

    # Load each stage, run postprocessing, validate strict alignment.
    for s in [1, 2, 3]:
        ds = CustomDataset(stage=s, force_reload=False, is_test=args.test)
        _validate_postprocessed(ds, s)

        # Print actual sample data (not just schema) for the first few items.
        print(f"\nStage {s} sample data (first 2 items):")
        for i in range(min(30,len(ds))):
            item = ds[i]
            if s in [1, 2]:
                # Embedding is a real tensor from GraphEmbedder
                key = next(iter(item["graph_labels"].keys()))
                emb = item["embeddings"]
                lbl = item["graph_labels"][key]
                emb_flat = emb.flatten()
                preview = emb_flat[:16].detach().cpu().tolist() if isinstance(emb, torch.Tensor) else []
                print(f"  [{i}] key={key}")
                if isinstance(emb, torch.Tensor):
                    print(
                        f"      embedding: shape={tuple(emb.shape)} preview[:16]={preview} mean={emb.mean().item():.6f} std={emb.std().item():.6f}"
                    )
                else:
                    print(f"      embedding: type={type(emb)}")
                print(f"      label: {float(lbl.detach().cpu().item()) if isinstance(lbl, torch.Tensor) else lbl}")
            else:
                g = item["graph"]
                cfg = item["cfg_labels"]
                ast = item["ast_labels"]
                print(f"  [{i}] project={item['project_name']} ntypes={list(g.ntypes)}")
                for ntype in g.ntypes:
                    n = g.num_nodes(ntype)
                    feat = g.nodes[ntype].data.get("feat")
                    feat_shape = tuple(feat.shape) if isinstance(feat, torch.Tensor) else None
                    feat_preview = (
                        feat[0, :8].detach().cpu().tolist()
                        if isinstance(feat, torch.Tensor) and feat.numel() > 0
                        else None
                    )
                    print(f"      {ntype}: nodes={n} feat_shape={feat_shape} feat0[:8]={feat_preview}")

                cfg_preview = cfg[:2].detach().cpu().tolist() if isinstance(cfg, torch.Tensor) else None
                ast_preview = ast[:2].detach().cpu().tolist() if isinstance(ast, torch.Tensor) else None
                print(f"      cfg_labels: shape={tuple(cfg.shape)} first2={cfg_preview}")
                print(f"      ast_labels: shape={tuple(ast.shape)} first2={ast_preview}")

    print("\n✅ All 3 stages are schema-consistent and postprocessing-aligned.")
