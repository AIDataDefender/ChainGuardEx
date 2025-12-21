from torch.utils.data import Dataset
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
from pathlib import Path

# Add the workspace root to Python path
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

try:
    from experiments.the_utils.logger import setup_logger
    from experiments.cpg_processor import CPG_Processor
    from experiments.the_utils import graph_utils as _graph_utils
except Exception:
    from the_utils.logger import setup_logger
    from cpg_processor import CPG_Processor
    from the_utils import graph_utils as _graph_utils

from Data.DAppSCAN.f6_DAppSCAN_fetch_data_adapter import f6_fetch_DAppSCAN_data
from f6_fetch_MANDO_data_adapter import f6_fetch_MANDO_data


# Logging Setup
log_folder = os.getenv("LOG_FOLDER", "Logs")
os.makedirs(log_folder, exist_ok=True)
logger = setup_logger(f"{log_folder}/Dataset.log")

os.environ["DGLBACKEND"] = "pytorch"
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def _rel_to_str_tuple(rel):
    """Convert a canonical etype tuple elements to strings (handles Enums)."""
    if not isinstance(rel, tuple) or len(rel) != 3:
        return rel
    out = []
    for x in rel:
        if hasattr(x, "value"):
            out.append(str(x.value))
        else:
            out.append(str(x))
    return tuple(out)


def standardize_heterograph(
    g,
    required_rel_names,
    embedding_dims,
    *,
    filter_out_node_types=None,
    filter_out_edge_types=None,
    drop_non_schema_match=False,
    report=False,
    fill_strategy="zeros",
    noise_std=0.02,
):
    """Produce a standardized heterograph with optional schema filtering.

    Default behavior (backwards-compatible):
    - Ensure presence of cfg_node and ast_node (0 nodes if missing).
    - Add all requested canonical edge types (0 edges if missing).
    - Populate node/edge 'feat' with copied tensors when available, otherwise zeros.

    Options:
    - filter_out_node_types: list[str]; remove these node types entirely.
      Edges whose src/dst node types are removed are also removed.
    - filter_out_edge_types: list[canonical_etype]; remove these canonical etypes.
      Accepts tuples like (src, etype, dst); elements are stringified.
    - drop_non_schema_match: if True AND no explicit filtering is enabled,
      return None when the input graph does not contain all required node/edge types.
    - report: print a short report of what was filtered and what remains.
    - fill_strategy for missing features (when node/edge exists but has no feat):
      'zeros' (default), 'mean' (mean-vector fill when possible), 'gaussian'
      (deterministic N(0, noise_std) based on type name).
    """
    if g is None or not hasattr(g, "ntypes") or not hasattr(g, "canonical_etypes"):
        return g

    def _report(msg: str):
        if report:
            try:
                print(msg)
            except Exception:
                pass

    # Normalize relation tuples to simple string triples
    def _to_rel(r):
        r = _rel_to_str_tuple(r)
        return r if (isinstance(r, tuple) and len(r) == 3) else None

    def _to_ntype(x):
        try:
            return str(getattr(x, "value", x))
        except Exception:
            return None

    filter_out_node_types = set(
        _to_ntype(x)
        for x in (filter_out_node_types or [])
        if _to_ntype(x)
    )
    filter_out_edge_types = set(
        r
        for r in (_to_rel(x) for x in (filter_out_edge_types or []))
        if r
    )
    filtering_enabled = bool(filter_out_node_types or filter_out_edge_types)

    try:
        req_rels = [r for r in (_to_rel(x)
                                for x in (required_rel_names or [])) if r]
    except Exception:
        req_rels = []

    if not req_rels:
        try:
            req_rels = [r for r in (_to_rel(x) for x in list(
                getattr(g, "canonical_etypes", []))) if r]
        except Exception:
            req_rels = []

    # When requested, drop graphs that don't already match schema (no padding/standardization)
    if drop_non_schema_match and not filtering_enabled:
        existing_str = set(_to_rel(x) for x in getattr(g, "canonical_etypes", []) or [])
        existing_str = set(x for x in existing_str if x)

        required_ntypes_check = set(["cfg_node", "ast_node"])
        for (s, _, d) in req_rels:
            required_ntypes_check.add(s)
            required_ntypes_check.add(d)

        missing_ntypes = sorted([nt for nt in required_ntypes_check if nt not in set(map(str, g.ntypes))])
        missing_rels = sorted([rel for rel in req_rels if rel not in existing_str])
        if missing_ntypes or missing_rels:
            _report(
                "[standardize_heterograph] Dropping non-schema-match graph: "
                f"missing_ntypes={missing_ntypes} missing_rels={missing_rels}"
            )
            return None

    required_ntypes = set(["cfg_node", "ast_node"])
    for (s, _, d) in req_rels:
        required_ntypes.add(s)
        required_ntypes.add(d)

    # Apply node-type filtering (remove node types entirely)
    if filter_out_node_types:
        required_ntypes -= set(filter_out_node_types)

    if not required_ntypes:
        _report(
            "[standardize_heterograph] All node types filtered out; returning None."
        )
        return None

    # Node counts (0 if missing)
    num_nodes_dict = {nt: int(g.num_nodes(nt)) if nt in g.ntypes else 0 for nt in sorted(required_ntypes)}

    # Build edge data mapping for heterograph creation
    data_dict = {}
    existing = set(getattr(g, "canonical_etypes", []))
    existing_str = set(_to_rel(x) for x in existing)
    existing_str = set(x for x in existing_str if x)

    removed_edges_by_filter = []
    kept_edges = 0
    for rel in req_rels:
        s, e, d = rel
        if rel in filter_out_edge_types:
            removed_edges_by_filter.append(rel)
            continue
        if s not in num_nodes_dict or d not in num_nodes_dict:
            continue
        if rel in existing_str:
            try:
                src, dst = g.edges(etype=rel)
            except Exception:
                src = torch.empty((0,), dtype=torch.int64)
                dst = torch.empty((0,), dtype=torch.int64)
        else:
            src = torch.empty((0,), dtype=torch.int64)
            dst = torch.empty((0,), dtype=torch.int64)
        data_dict[rel] = (src, dst)
        kept_edges += 1

    # Ensure at least one relation exists for DGL heterograph creation
    if not data_dict:
        dummy_rel = ("cfg_node", "_dummy", "cfg_node")
        data_dict[dummy_rel] = (torch.empty(
            (0,), dtype=torch.int64), torch.empty((0,), dtype=torch.int64))

    new_g = dgl.heterograph(data_dict, num_nodes_dict=num_nodes_dict)

    if report:
        try:
            removed_ntypes = sorted(list(filter_out_node_types))
        except Exception:
            removed_ntypes = []
        _report(
            "[standardize_heterograph] "
            f"removed_ntypes={removed_ntypes} removed_etypes={len(removed_edges_by_filter)} "
            f"kept_ntypes={len(new_g.ntypes)} kept_etypes={kept_edges} "
            f"total_nodes={sum(int(new_g.num_nodes(nt)) for nt in new_g.ntypes)} "
            f"total_edges={sum(int(new_g.num_edges(et)) for et in new_g.canonical_etypes)}"
        )

    # Preserve id maps
    for attr in ("contract_id_to_name", "function_id_to_name"):
        if hasattr(g, attr):
            try:
                setattr(new_g, attr, getattr(g, attr))
            except Exception:
                pass

    cfg_dim = int(embedding_dims.get("cfg_node", 0) or 0)
    ast_dim = int(embedding_dims.get("ast_node", 0) or 0)
    edge_dim = int(embedding_dims.get("edge", 0) or 0)

    def _make_fill(shape, *, key: str, strategy: str, mean_vec=None):
        strategy = (strategy or "zeros").lower().strip()
        if strategy == "zeros":
            return torch.zeros(shape, dtype=torch.float32)
        if strategy == "mean" and isinstance(mean_vec, torch.Tensor) and mean_vec.numel() > 0:
            if mean_vec.dim() == 1 and len(shape) == 2 and mean_vec.shape[0] == shape[1]:
                return mean_vec.to(torch.float32).unsqueeze(0).expand(shape[0], shape[1]).contiguous()
            # Fallback if mean vector shape mismatches
        if strategy == "gaussian":
            gen = torch.Generator(device="cpu")
            gen.manual_seed(42)
            return (torch.randn(shape, generator=gen, dtype=torch.float32) * float(noise_std)).to(torch.float32)
        # Default fallback
        return torch.zeros(shape, dtype=torch.float32)

    # Node features: copy when possible, otherwise create zeros (stable shapes)
    for ntype in new_g.ntypes:
        n = int(new_g.num_nodes(ntype))
        if n == 0:
            if ntype == "cfg_node" and cfg_dim > 0:
                new_g.nodes[ntype].data["feat"] = torch.zeros(
                    (0, cfg_dim), dtype=torch.float32)
            elif ntype == "ast_node" and ast_dim > 0:
                new_g.nodes[ntype].data["feat"] = torch.zeros(
                    (0, ast_dim), dtype=torch.float32)
            continue

        feat = None
        if ntype in g.ntypes:
            try:
                feat = g.nodes[ntype].data.get("feat")
            except Exception:
                feat = None
        if isinstance(feat, torch.Tensor) and feat.shape[0] == n:
            new_g.nodes[ntype].data["feat"] = feat.to(torch.float32)
        else:
            # Prefer known dims; otherwise infer from existing tensor when possible
            dim = cfg_dim if ntype == "cfg_node" else (ast_dim if ntype == "ast_node" else 0)
            if dim <= 0 and isinstance(feat, torch.Tensor) and feat.dim() == 2:
                dim = int(feat.shape[1])

            if dim > 0:
                mean_vec = None
                if fill_strategy == "mean" and isinstance(feat, torch.Tensor) and feat.dim() == 2 and feat.shape[1] == dim and feat.numel() > 0:
                    mean_vec = feat.to(torch.float32).mean(dim=0)
                new_g.nodes[ntype].data["feat"] = _make_fill(
                    (n, dim),
                    key=f"node:{ntype}:{dim}",
                    strategy=fill_strategy,
                    mean_vec=mean_vec,
                )

    # Edge features: copy when possible, otherwise zeros
    for rel in new_g.canonical_etypes:
        m = int(new_g.num_edges(rel))
        if m == 0:
            continue
        feat = None
        if rel in existing_str:
            try:
                feat = g.edges[rel].data.get("feat")
            except Exception:
                feat = None
        if isinstance(feat, torch.Tensor) and feat.shape[0] == m:
            new_g.edges[rel].data["feat"] = feat.to(torch.float32)
        else:
            dim = edge_dim
            if dim <= 0 and isinstance(feat, torch.Tensor) and feat.dim() == 2:
                dim = int(feat.shape[1])

            if dim > 0:
                mean_vec = None
                if fill_strategy == "mean" and isinstance(feat, torch.Tensor) and feat.dim() == 2 and feat.shape[1] == dim and feat.numel() > 0:
                    mean_vec = feat.to(torch.float32).mean(dim=0)
                rel_key = ":".join(map(str, _rel_to_str_tuple(rel)))
                new_g.edges[rel].data["feat"] = _make_fill(
                    (m, dim),
                    key=f"edge:{rel_key}:{dim}",
                    strategy=fill_strategy,
                    mean_vec=mean_vec,
                )

    return new_g


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
        load_dir="./save_data",
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
        # Stage 1/2 optional embedding cache (loaded on-demand in postprocessing).
        self.dataset_embedding = {}
        self.is_test = is_test

        # Initialize Processor components only if needed (Force Reload or Missing Data)
        # We check simple existence first to avoid loading heavy models unnecessarily
        self.base_filename = f"{source}_dataset.pt"
        self.save_dir = "./save_data"
        os.makedirs(self.save_dir, exist_ok=True)
        self.load_dir = load_dir
        # Check if we need to process
        needs_processing = force_reload
        if not needs_processing:
            # Check if specific stage files exist (graph + label)
            test_suffix = "test" if self.is_test else ""
            g_path = os.path.join(
                self.load_dir,
                self.base_filename.replace(
                    ".pt", f"_stage{stage}{test_suffix}_graph.pt"),
            )
            l_path = os.path.join(
                self.load_dir,
                self.base_filename.replace(
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
            batch_size=512,
            checkpoint_dir="./checkpoints/processed_graphs",
        )
        self.embedding_dims = self.CPG_Proccessor.embedding_dims

        node_in_dims = {}
        if "cfg_node" in self.embedding_dims:
            node_in_dims["cfg_node"] = int(self.embedding_dims["cfg_node"])
        if "ast_node" in self.embedding_dims:
            node_in_dims["ast_node"] = int(self.embedding_dims["ast_node"])

        print(f"  Embedding Dims: {json.dumps(self.embedding_dims, indent=2)}")
        if source == "DAppSCAN":
            self._fetch_and_process_DAppSCAN_data(
                force_reload=needs_processing)
        elif source == "MANDO":
            self._fetch_and_process_MANDO_data(
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
        base_filename = self.base_filename
        saved_file_path = os.path.join(self.load_dir, base_filename)
        load_file_path = os.path.join(self.load_dir, base_filename) 
        try:
            # 1. Try Loading Existing Data (If not forced)
            if not force_reload:
                if self._load_data(load_file_path):
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

    def _fetch_and_process_MANDO_data(self, force_reload=False):
        """Orchestrates loading from disk or fetching & processing data."""
        base_filename = self.base_filename
        saved_file_path = os.path.join(self.save_dir, base_filename)
        load_file_path = os.path.join(self.load_dir, base_filename) 
        try:
            # 1. Try Loading Existing Data (If not forced)
            if not force_reload:
                if self._load_data(load_file_path):
                    return

            # 2. Process New Data (Generates ALL stages)
            logger.info("Fetching and processing MANDO data...")
            res = f6_fetch_MANDO_data(
                root=Path("/mnt/d/KLTN2/DatasetEtherScanio/ge-sc-data/ProcessedData/success/").as_posix() , is_test=self.is_test
            )

            if not res or not res[0] or not res[1]:
                logger.error("Failed to fetch MANDO data.")
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
            os.makedirs(os.path.dirname(base_path), exist_ok=True)
            logger.info("Saving 3-stage cascade data to disk...")
            for s in [1, 2, 3]:
                key = f"stage{s}"
                print(f"  Saving Stage {s} data...")
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
                            print(
                                f"Stage {s} graphs for {p_name} is not a dict; skipping"
                            )
                            continue
                        for subkey, g in graph_dict.items():
                            flattened_graphs.append((p_name, subkey, g))
                    for p_name, label_dict in data.get("labels", []):
                        if not isinstance(label_dict, dict):
                            print(
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
        # Paths
        test_suffix = "test" if self.is_test else ""
        g_path = base_path.replace(
            ".pt", f"_stage{s}{test_suffix}_graph.pt"
        )
        l_path = base_path.replace(
            ".pt", f"_stage{s}{test_suffix}_label.pt"
        )

        if os.path.exists(g_path) and os.path.exists(l_path):
            logger.info(f"Loading Stage {s} from disk...")
            flattened_graphs = torch.load(
                g_path, map_location="cpu", weights_only=False, mmap=True
            )
            flattened_labels = torch.load(
                l_path, map_location="cpu", weights_only=False, mmap=True
            )

            # Deflatten: Group by project
            graphs_by_project = {}
            for item in flattened_graphs:
                if isinstance(item, (list, tuple)) and len(item) == 3:
                    p_name, subkey, g = item
                else:
                    logger.warning(
                        f"Unexpected graph item format: {type(item)}")
                    continue
                graphs_by_project.setdefault(p_name, {})[subkey] = g

            labels_by_project = {}
            for item in flattened_labels:
                if isinstance(item, (list, tuple)) and len(item) == 3:
                    p_name, subkey, lbl = item
                else:
                    logger.warning(
                        f"Unexpected label item format: {type(item)}")
                    continue
                labels_by_project.setdefault(p_name, {})[subkey] = lbl

            # Store as dicts keyed by project to guarantee correct alignment in postprocessing.
            self.dataset_graph[s] = graphs_by_project
            self.dataset_label[s] = labels_by_project

            # Collect rel_names from loaded graphs
            all_rels = set()
            for p, gdict in graphs_by_project.items():
                for sub, g in gdict.items():
                    if hasattr(g, 'canonical_etypes'):
                        all_rels.update(g.canonical_etypes)
            self.CPG_Proccessor.rel_names = list(all_rels)

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
        graphs_by_project = dict(raw_graphs) if isinstance(
            raw_graphs, list) else raw_graphs
        labels_by_project = dict(raw_labels) if isinstance(
            raw_labels, list) else raw_labels

        if not isinstance(graphs_by_project, dict) or not isinstance(labels_by_project, dict):
            logger.error(
                f"Unexpected stage containers: graphs={type(graphs_by_project)} labels={type(labels_by_project)}"
            )
            return

        total_items = 0
        for _p, _gdict in graphs_by_project.items():
            if isinstance(_gdict, dict):
                total_items += len(_gdict)

        pbar = None
        if total_items > 0:
            pbar = tqdm(
                total=total_items,
                desc=f"Postprocessing Stage {self.stage}",
                unit="item",
            )

        try:
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

                    if self.stage in [1, 2]:
                        g = graph_data

                        if not hasattr(g, "ntypes"):
                            raise TypeError(
                                f"Stage 1/2 expects a DGL heterograph, got {type(g)} for {full_key}")


                        if not isinstance(lbl, int) or lbl not in (0, 1):
                            raise ValueError(
                                f"Stage {self.stage} label must be int 0/1 for {full_key}, got {lbl} ({type(lbl)})"
                            )
                    else:
                        g = graph_data
                        # Stage 3: fail-fast contract check: label rows must match node counts.
                        if not hasattr(g, "ntypes"):
                            raise TypeError(
                                f"Stage 3 expects a DGL heterograph, got {type(g)} for {full_key}")

                        if not isinstance(lbl, dict) or ("cfg_node" not in lbl or "ast_node" not in lbl):
                            raise TypeError(
                                f"Stage 3 label must be dict with cfg_node/ast_node for {full_key}, got {type(lbl)}"
                            )

                        expected_dim = len(getattr(_graph_utils, "OWASP_VULN", []))


                        cfg_nodes = g.num_nodes("cfg_node") if "cfg_node" in g.ntypes else 0
                        ast_nodes = g.num_nodes("ast_node") if "ast_node" in g.ntypes else 0

                        cfg_rows = int(lbl.get("cfg_node", []) and len(
                            lbl.get("cfg_node", [])) or 0)
                        ast_rows = int(lbl.get("ast_node", []) and len(
                            lbl.get("ast_node", [])) or 0)
                        
                        if cfg_rows != cfg_nodes:
                            raise ValueError(
                                f"Stage 3 contract mismatch for CFG {full_key}"
                            )
                        if ast_rows != ast_nodes:
                            raise ValueError(
                                f"Stage 3 contract mismatch for AST {full_key}"
                            )

                        # Standardize heterograph schema so batching never drops data.
                        try:
                            g = standardize_heterograph(
                                g,
                                getattr(self.CPG_Proccessor,
                                        "rel_names", None),
                                self.embedding_dims,
                            )
                        except Exception as e:
                            logger.warning(
                                f"Stage 3 schema standardization failed for {full_key}: {e}")
                            if pbar is not None:
                                pbar.update(1)
                            continue

                        # Materialize label tensors with stable 2D shape [N, expected_dim]
                        # (torch.tensor([]) would otherwise create a 1D empty tensor).
                        cfg_list = lbl.get("cfg_node", []) or []
                        ast_list = lbl.get("ast_node", []) or []
                        cfg_tensor = torch.tensor(
                            cfg_list, dtype=torch.float32)
                        ast_tensor = torch.tensor(
                            ast_list, dtype=torch.float32)

                        if cfg_tensor.numel() == 0:
                            cfg_tensor = torch.zeros(
                                (0, expected_dim), dtype=torch.float32)
                        elif cfg_tensor.dim() == 1 and cfg_tensor.numel() == expected_dim:
                            cfg_tensor = cfg_tensor.unsqueeze(0)

                        if ast_tensor.numel() == 0:
                            ast_tensor = torch.zeros(
                                (0, expected_dim), dtype=torch.float32)
                        elif ast_tensor.dim() == 1 and ast_tensor.numel() == expected_dim:
                            ast_tensor = ast_tensor.unsqueeze(0)


                    if self.stage in [1, 2]:
                        g = standardize_heterograph(
                            g,
                            getattr(self.CPG_Proccessor, "rel_names", None),
                            self.embedding_dims,
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
                                    "cfg_node": cfg_tensor,
                                    "ast_node": ast_tensor,
                                },
                            )
                        )

                    if pbar is not None:
                        pbar.update(1)

        finally:
            if pbar is not None:
                pbar.close()

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
            return {"graph": g, "graph_labels": {k: label}, "project_name": p_name}
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

        # 2. Batch Graphs
        batched_graph = None
        kept_batch = batch


        graphs = [b["graph"] for b in batch if isinstance(
            b, dict) and b.get("graph") is not None]
        if not graphs:
            return None

        # All Stage 3 graphs should have been standardized during dataset postprocessing.
        batched_graph = dgl.batch(graphs)

        # Preserve metadata
        id_to_name_maps = []
        for g in graphs:
            contract_map = getattr(g, "contract_id_to_name", None)
            function_map = getattr(g, "function_id_to_name", None)
            id_to_name_maps.append(
                {"contract": contract_map, "function": function_map})
        batched_graph.id_to_name_maps = id_to_name_maps

        # 3. Collate Labels
        def collate_key(key, items_batch):
            items = [b.get(key) for b in items_batch if b.get(key) is not None]

            # A. Tensor Mode (Stage 3 Node Labels) -> Concat
            if all(isinstance(x, torch.Tensor) for x in items):
                return torch.cat(items, dim=0) if items else None

            # B. Dict/Raw Mode (Stage 1/2 Labels) -> List
            return items

        # Stage-specific collation
        if stage == 3:
            labels = {
                "cfg_labels": collate_key("cfg_labels", kept_batch),
                "ast_labels": collate_key("ast_labels", kept_batch),
            }
        else:
            labels = {"graph_labels": [b.get("graph_labels") for b in batch]}

        return {
            "graph": batched_graph,
            **labels,
            "project_names": [b["project_name"] for b in (kept_batch if stage == 3 else batch)],
        }
    except Exception as e:
        logger.error(f"Error in custom_collate: {e}")
        return None











########################################################
##################### TEST CODE ########################
########################################################
if __name__ == "__main__":
    # Command Line Interface for Data Processing
    parser = argparse.ArgumentParser(description="DAppSCAN Dataset Processor")
    parser.add_argument(
        "--force_reload",
        action="store_true",
        help="If set, fully re-processes raw data and regenerates graphs for all 3 stages.",
    )
    parser.add_argument(
        "--source",
        type=str,
        default="DAppSCAN",
        choices=["DAppSCAN", "MANDO"],
        help="Which source to load",
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


    # One processing pass generates and saves all 3 stages.
    _ = CustomDataset( source=args.source,
        stage=3, force_reload=args.force_reload, is_test=args.test)
