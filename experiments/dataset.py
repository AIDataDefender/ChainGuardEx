import logging
from torch.utils.data import Dataset
from tqdm import tqdm
import torch
import numpy as np
import random
import json
import sys
import os
import traceback
import dgl
import argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

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


# Logging Setup
log_folder = os.getenv("LOG_FOLDER", "Logs")
os.makedirs(log_folder, exist_ok=True)
logger = setup_logger(f"{log_folder}/Dataset.log", logging.INFO)

os.environ["DGLBACKEND"] = "pytorch"
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def _print_path_banner(title: str, rows):
    """Pretty-print a decorated block for IO paths/settings."""
    try:
        title = str(title)
        rows = list(rows or [])
        deco = "═" * 14
        header = f"{deco} {title} {deco}"
        print("\n" + header)
        for k, v in rows:
            print(f"  ▸ {k}: {v}")
        print("═" * len(header) + "\n")
    except Exception:
        # Never let debug printing break training runs.
        pass


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
        existing_str = set(_to_rel(x)
                           for x in getattr(g, "canonical_etypes", []) or [])
        existing_str = set(x for x in existing_str if x)

        required_ntypes_check = set(["cfg_node", "ast_node"])
        for (s, _, d) in req_rels:
            required_ntypes_check.add(s)
            required_ntypes_check.add(d)

        missing_ntypes = sorted(
            [nt for nt in required_ntypes_check if nt not in set(map(str, g.ntypes))])
        missing_rels = sorted(
            [rel for rel in req_rels if rel not in existing_str])
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
    num_nodes_dict = {nt: int(g.num_nodes(
        nt)) if nt in g.ntypes else 0 for nt in sorted(required_ntypes)}

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
            dim = cfg_dim if ntype == "cfg_node" else (
                ast_dim if ntype == "ast_node" else 0)
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


class PretrainConfig:
    class CodeBERT:
        name = "microsoft/codebert-base"
        embedding_dim = 768

    class E5:
        name = "intfloat/e5-base-v2"
        embedding_dim = 768

    class MiniLM:
        name = "sentence-transformers/all-MiniLM-L6-v2"
        embedding_dim = 384

    class CodeT5_100M:
        name = "Salesforce/codet5p-110m-embedding"
        embedding_dim = 256


class CustomDataset(Dataset):
    """
    3-Stage Cascaded Dataset.
    Stage 1: Contract-Level (Binary)
    Stage 2: Function-Level (Binary)
    Stage 3: Block-Level (Multi-label Node Classification)
    """

    # Default dataset shard directories per source (Windows paths).
    # These are ONLY for dataset shards (stage graph/label files), not checkpoints.
    DEF_DATA_LOAD_DIR = {
        "DAppSCAN": r"/mnt/d/KLTN2/save_data2",
        "MANDO": r"/mnt/d/KLTN2/save_data3",
        "EtherScanIO": r"./save_data",
    }

    DEF_DATA_CHECKPOINT_DIR = {
        "DAppSCAN": r"/mnt/d/KLTN2/DatasetEtherScanio/DAppSCAN/checkpoints/processed_graphs",
        "MANDO": r"/mnt/d/KLTN2/DatasetEtherScanio/MANDO/checkpoints/processed_graphs",
        "EtherScanIO": r"/mnt/d/KLTN2/DatasetEtherScanio/EtherScanio/checkpoints/processed_graphs",
    }

    def __init__(
        self,
        source="DAppSCAN",
        load_dir=None,
        force_reload=False,
        rand_seed=42,
        stage=3,
        is_test=False,
        embedding_model_conf=PretrainConfig.CodeBERT,
        # embedding_model_name="intfloat/e5-base-v2",

    ):
        logger.info(f"Initializing CustomDataset for Stage {stage}...")
        self.stage = stage
        if stage not in [1, 2, 3]:
            raise ValueError(f"Invalid stage {stage}. Must be 1, 2, or 3.")

        self._set_rand_seed(rand_seed)
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.embedding_model__name_id = embedding_model_conf.name
        self.embedding_model__name_id = self.embedding_model__name_id.replace(
            "/", "_")

        self.dataset_graph = {}
        self.dataset_label = {}
        # Stage 1/2 optional embedding cache (loaded on-demand in postprocessing).
        self.dataset_embedding = {}
        self.is_test = is_test
        self.temp_files = []

        # Initialize Processor components only if needed (Force Reload or Missing Data)
        # We check simple existence first to avoid loading heavy models unnecessarily
        self.base_filename = f"{source}_dataset.pt"

        # Resolve default load/save dirs from the static mapping unless explicitly provided.
        resolved_load_dir = load_dir
        if resolved_load_dir in (None, "", "./save_data"):
            resolved_load_dir = self.DEF_DATA_LOAD_DIR.get(
                source, "./save_data")
        self.load_dir = resolved_load_dir

        resolved_save_dir = self.DEF_DATA_LOAD_DIR.get(
            source, self.load_dir)
        self.save_dir = resolved_save_dir
        os.makedirs(self.save_dir, exist_ok=True)

        # IO banner early (helps confirm where shards/checkpoints go).
        _print_path_banner(
            "DATASET IO (INIT)",
            [
                ("source", source),
                ("stage", stage),
                ("is_test", self.is_test),
                ("embedding", self.embedding_model__name_id),
                ("load_dir", os.path.abspath(self.load_dir)),
                ("save_dir (shards)", os.path.abspath(self.save_dir)),
                ("base_filename", self.base_filename),
            ],
        )
        # Check if we need to process
        needs_processing = force_reload
        if not needs_processing:
            # Check if specific stage files exist (graph + label) - supports both single files and shards.
            found_pair = self._find_existing_stage_pair(
                os.path.join(self.load_dir, self.base_filename), stage
            )
            if found_pair is None:
                needs_processing = True
                logger.info(
                    f"[YELLOW] Stage {stage} data not found. Triggering processing.")

        self.tokenizer = None
        self.embedding_model = None
        if needs_processing:
            from transformers import AutoModel, AutoTokenizer

            logger.info(
                f"Loading {embedding_model_conf.name} for data processing...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                embedding_model_conf.name, use_fast=True
            )
            self.embedding_model = AutoModel.from_pretrained(
                embedding_model_conf.name)
        else:
            logger.info("Loading mode: Skipping CodeBERT initialization.")

        self.CPG_Proccessor = CPG_Processor(
            tokenizer=self.tokenizer,
            model=self.embedding_model,
            device=self.device,
            batch_size=512,
            checkpoint_dir="/mnt/d/KLTN2/DatasetEtherScanio/EtherScanio/checkpoints/processed_graphs",
            embedding_model_conf=embedding_model_conf,
        )

        # Checkpoint dir banner (do not modify checkpoint_dir; only report it)
        _print_path_banner(
            "CHECKPOINT IO",
            [("checkpoint_dir", str(
                getattr(self.CPG_Proccessor, "checkpoint_dir", "<missing>")))],
        )
        self.rel_names = None
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
        elif source == "EtherScanIO":
            self._fetch_and_process_EtherScanIO_data(
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

    def _fetch_and_process_generic(self, *, source_name: str, fetch_fn, root, force_reload: bool = False):
        """Shared fetch/process/save/load pipeline for all sources.

        Keeps logic consistent across DAppSCAN/MANDO/EtherScanIO.
        """
        base_filename = self.base_filename
        saved_file_path = os.path.join(self.save_dir, base_filename)
        load_file_path = os.path.join(self.load_dir, base_filename)

        _print_path_banner(
            f"FETCH+PROCESS ({source_name})",
            [
                ("root", str(root)),
                ("saved_file_base", os.path.abspath(saved_file_path)),
                ("load_file_base", os.path.abspath(load_file_path)),
            ],
        )

        try:
            # 1. Try Loading Existing Data (If not forced)
            if not force_reload:
                if self._load_data(load_file_path):
                    return

            # 2. Process New Data (Generates ALL stages)
            logger.info(f"Fetching and processing {source_name} data...")
            res = fetch_fn(root=root, is_test=self.is_test)

            if not res or not res[0] or not res[1]:
                logger.error(f"Failed to fetch {source_name} data.")
                sys.exit(1)

            cpg_list, vuln_json_list = res

            # 3. Run 3-Stage Processing (writes per-project checkpoints)
            logger.info("Running CPG Processor (generating all 3 stages)...")
            selected_projects = self.CPG_Proccessor.process_graphs(
                cpg_list, vuln_json_list)

            # 4. Save all stages from per-project checkpoints (no temp files)
            self._save_data_from_checkpoints(
                saved_file_path, allowed_projects=selected_projects)

            # 5. Load the saved data
            self._load_data(load_file_path)

        except Exception as e:
            logger.error(f"Error in data processing ({source_name}): {e}")
            traceback.print_exc()

    def _fetch_and_process_DAppSCAN_data(self, force_reload=False):
        from Data.DAppSCAN.f6_DAppSCAN_fetch_graph_data_adapter import f6_fetch_DAppSCAN_data
        return self._fetch_and_process_generic(
            source_name="DAppSCAN",
            fetch_fn=f6_fetch_DAppSCAN_data,
            root=r"Data/DAppSCAN/ProcessedData/success",
            force_reload=force_reload,
        )

    def _fetch_and_process_MANDO_data(self, force_reload=False):
        from experiments.f6_MANDO_fetch_graph_data_adapter import f6_fetch_MANDO_data
        return self._fetch_and_process_generic(
            source_name="MANDO",
            fetch_fn=f6_fetch_MANDO_data,
            root=Path(
                "/mnt/d/KLTN2/DatasetEtherScanio/ge-sc-data/ProcessedData/success/").as_posix(),
            force_reload=force_reload,
        )

    def _fetch_and_process_EtherScanIO_data(self, force_reload=False):
        from experiments.f6_EtherScanIO_fetch_graph_data_adapter import f6_fetch_EtherScanIO_data
        return self._fetch_and_process_generic(
            source_name="EtherScanIO",
            fetch_fn=f6_fetch_EtherScanIO_data,
            root=Path(
                "/mnt/d/KLTN2/DatasetEtherScanio/EtherScanio/ProcessedData/success/").as_posix(),
            force_reload=force_reload,
        )

    def _stage_paths_candidates(self, base_path: str, stage: int, artifact: str):
        """Return ordered candidate base paths for a stage artifact.

        artifact: 'graph' or 'label'

        Supports legacy naming that used 'test' without an underscore separator.
        """
        test_tag_legacy = "test" if self.is_test else ""
        test_tag_preferred = "test_" if self.is_test else ""

        # Preferred: stage{n}_test_<embedding>_graph.pt
        preferred = base_path.replace(
            ".pt",
            f"_stage{stage}_{test_tag_preferred}{self.embedding_model__name_id}_{artifact}.pt",
        )

        # Legacy: stage{n}_test<embedding>_graph.pt
        legacy = base_path.replace(
            ".pt",
            f"_stage{stage}_{test_tag_legacy}{self.embedding_model__name_id}_{artifact}.pt",
        )

        # De-duplicate while preserving order
        out = []
        for p in (preferred, legacy):
            if p not in out:
                out.append(p)
        return out

    def _find_existing_stage_pair(self, base_path: str, stage: int):
        """Find a matching (graph_base, label_base) pair for the given stage.

        Returns:
            tuple[str, str] if a usable pair exists (single or sharded), else None.
        """
        import glob

        graph_candidates = self._stage_paths_candidates(
            base_path, stage, "graph")
        label_candidates = self._stage_paths_candidates(
            base_path, stage, "label")

        for g_path in graph_candidates:
            for l_path in label_candidates:
                # Single-file pair
                if os.path.exists(g_path) and os.path.exists(l_path):
                    return (g_path, l_path)

                # Sharded pair (require at least shard 0 for both)
                g0 = g_path.replace(".pt", "_0.pt")
                l0 = l_path.replace(".pt", "_0.pt")
                if os.path.exists(g0) and os.path.exists(l0):
                    return (g_path, l_path)

                # If one side has shards but the other doesn't, don't accept it.
                has_g = bool(glob.glob(g_path.replace(".pt", "_*.pt")))
                has_l = bool(glob.glob(l_path.replace(".pt", "_*.pt")))
                if has_g != has_l:
                    continue

        return None

    def _clear_existing_shards(self, base_path: str, stage: int):
        import glob

        g_path = self._stage_paths_candidates(base_path, stage, "graph")[0]
        l_path = self._stage_paths_candidates(base_path, stage, "label")[0]
        for pat in (g_path.replace(".pt", "_*.pt"), l_path.replace(".pt", "_*.pt")):
            for fp in glob.glob(pat):
                try:
                    os.remove(fp)
                except Exception:
                    pass

    def _save_data_from_checkpoints(self, base_path: str, *, allowed_projects=None):
        """Create stage shards directly from per-project checkpoints.

        This removes the temp-file aggregation step and keeps peak memory bounded by
        flushing shard buffers in fixed-size item batches (default: 500 items per shard).

        If allowed_projects is provided, only checkpoints for those projects are
        included (this ensures --test mode only saves the fetched subset).
        """
        try:
            base_dir = os.path.dirname(base_path)
            if base_dir:
                os.makedirs(base_dir, exist_ok=True)

            # Performance knobs (env vars)
            # - MAX_ITEMS_PER_SHARD: flush after this many items (default: 500)
            # - SHARD_GC_EVERY: call gc.collect() every N checkpoints (default: 100)
            # - SHARD_STAGES: comma-separated stages to shard (default: 1,2,3)
            max_items_per_shard = int(os.getenv("MAX_ITEMS_PER_SHARD", "500"))
            shard_gc_every = int(os.getenv("SHARD_GC_EVERY", "100"))
            if shard_gc_every <= 0:
                shard_gc_every = 100
            try:
                shard_stages_raw = str(os.getenv("SHARD_STAGES", "1,2,3"))
                shard_stages = [int(s.strip()) for s in shard_stages_raw.split(",") if s.strip()]
                shard_stages = [s for s in shard_stages if s in (1, 2, 3)]
            except Exception:
                shard_stages = [1, 2, 3]
            if not shard_stages:
                shard_stages = [1, 2, 3]

            # Show where shards will be written (base_path is in save_dir).
            try:
                g3 = self._stage_paths_candidates(base_path, 3, "graph")[0]
                l3 = self._stage_paths_candidates(base_path, 3, "label")[0]
                _print_path_banner(
                    "SHARD SAVE (BASES)",
                    [
                        ("base_path", os.path.abspath(base_path)),
                        ("stage3_graph_base", os.path.abspath(g3)),
                        ("stage3_label_base", os.path.abspath(l3)),
                        ("MAX_ITEMS_PER_SHARD", os.getenv("MAX_ITEMS_PER_SHARD", "500")),
                        ("SHARD_GC_EVERY", shard_gc_every),
                        ("SHARD_STAGES", ",".join(str(s) for s in shard_stages)),
                        (
                            "allowed_projects",
                            f"{len(allowed_projects) if allowed_projects else 0} (0 means no filter)",
                        ),
                    ],
                )
            except Exception:
                pass
            checkpoint_dir = getattr(
                self.CPG_Proccessor, "checkpoint_dir", None)
            if checkpoint_dir is None:
                raise RuntimeError("CPG_Processor.checkpoint_dir is not set")

            checkpoint_dir = Path(checkpoint_dir)
            # When sharding after a fetch() run, only include checkpoints for
            # the projects the adapter selected (esp. in test mode).
            ckpts = []
            allow = None
            if allowed_projects:
                try:
                    allow = set(str(x) for x in allowed_projects if x)
                except Exception:
                    allow = None

            if allow:
                get_ckpt_path = getattr(
                    self.CPG_Proccessor, "_get_checkpoint_path", None)
                if callable(get_ckpt_path):
                    ckpt_map = {p: Path(str(get_ckpt_path(p)))
                                for p in sorted(allow)}
                else:
                    sanitize_any = getattr(
                        self.CPG_Proccessor, "_sanitize_name", None)
                    if callable(sanitize_any):
                        def _sanitize(name: str) -> str:
                            return str(sanitize_any(name))
                    else:
                        def _sanitize(name: str) -> str:
                            return "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in str(name))

                    ckpt_map = {p: (checkpoint_dir / f"{_sanitize(p)}.pt")
                                for p in sorted(allow)}

                ckpts = [path for path in ckpt_map.values() if path.exists()]
                missing = [p for p, path in ckpt_map.items()
                           if not path.exists()]
                if missing:
                    logger.warning(
                        f"Only found {len(ckpts)}/{len(allow)} selected checkpoints in {checkpoint_dir}; missing examples: {missing[:5]}"
                    )
            else:
                ckpts = sorted(list(checkpoint_dir.glob("*.pt")))

            if not ckpts:
                raise FileNotFoundError(
                    f"No checkpoint files found in {checkpoint_dir}")

            _print_path_banner(
                "SHARD INPUT (CHECKPOINTS)",
                [
                    ("checkpoint_dir", str(checkpoint_dir)),
                    ("checkpoint_files", len(ckpts)),
                ],
            )

            logger.info(
                f"Sharding from {len(ckpts)} checkpoints in {checkpoint_dir}...")

            # Fresh write: remove existing shards for this embedding/stage naming.
            for s in shard_stages:
                self._clear_existing_shards(base_path, s)

            shard_idx = {s: 0 for s in shard_stages}
            buf_g = {s: [] for s in shard_stages}
            buf_l = {s: [] for s in shard_stages}
            total_items = {s: 0 for s in shard_stages}

            def _flush(stage: int):
                if not buf_g[stage]:
                    return
                if len(buf_g[stage]) != len(buf_l[stage]):
                    raise ValueError(
                        f"Flush stage{stage}: buffer size mismatch graphs={len(buf_g[stage])} labels={len(buf_l[stage])}"
                    )
                g_base = self._stage_paths_candidates(
                    base_path, stage, "graph")[0]
                l_base = self._stage_paths_candidates(
                    base_path, stage, "label")[0]
                idx = shard_idx[stage]
                g_out = g_base.replace(".pt", f"_{idx}.pt")
                l_out = l_base.replace(".pt", f"_{idx}.pt")

                n_items = len(buf_g[stage])
                total_items[stage] += n_items

                _print_path_banner(
                    f"SHARD WRITE stage{stage} #{idx}",
                    [
                        ("items(graphs)", n_items),
                        ("items(labels)", len(buf_l[stage])),
                        ("graph_out", os.path.abspath(g_out)),
                        ("label_out", os.path.abspath(l_out)),
                    ],
                )
                logger.info(
                    f"[shard_write] stage={stage} shard={idx} items={n_items} graph_out={g_out} label_out={l_out}"
                )

                torch.save(buf_g[stage], g_out)
                torch.save(buf_l[stage], l_out)
                shard_idx[stage] += 1
                buf_g[stage].clear()
                buf_l[stage].clear()

            for i, ckpt_path in enumerate(tqdm(ckpts, desc="Sharding checkpoints"), start=1):
                ckpt = torch.load(
                    ckpt_path, map_location="cpu", weights_only=False)
                if not isinstance(ckpt, dict):
                    continue

                p_name = ckpt.get("project_name")
                if not p_name:
                    continue

                for s in shard_stages:
                    g_dict = ckpt.get(f"stage{s}_graph")
                    l_dict = ckpt.get(f"stage{s}_labels")
                    if not isinstance(g_dict, dict) or not isinstance(l_dict, dict):
                        continue

                    keys = sorted(list(g_dict.keys()))
                    if set(keys) != set(l_dict.keys()):
                        raise ValueError(
                            f"Checkpoint {ckpt_path.name} stage{s}: graph/label keys mismatch"
                        )

                    for subkey in keys:
                        g = g_dict[subkey]
                        lbl = l_dict[subkey]
                        buf_g[s].append((p_name, subkey, g))
                        buf_l[s].append((p_name, subkey, lbl))

                        # Flush purely by item count (fast; avoids per-tensor scans).
                        if len(buf_g[s]) >= max_items_per_shard:
                            _flush(s)

                # Free checkpoint quickly
                del ckpt
                # Full GC is expensive; do it periodically instead of per checkpoint.
                if i % shard_gc_every == 0:
                    import gc
                    gc.collect()

            for s in shard_stages:
                _flush(s)

            _print_path_banner(
                "SHARD SUMMARY",
                [
                    ("shard_stages", ",".join(str(s) for s in shard_stages)),
                    *[(f"stage{s}_total_items", total_items.get(s, 0)) for s in shard_stages],
                    *[(f"stage{s}_shards", shard_idx.get(s, 0)) for s in shard_stages],
                ],
            )
            logger.info(
                f"[shard_summary] stages={shard_stages} items={total_items} shards={shard_idx}"
            )

            import gc
            gc.collect()
            if torch.cuda.is_available():
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass
            logger.info("Checkpoint sharding complete.")
        except Exception as e:
            logger.error(f"Failed to shard from checkpoints: {e}")
            traceback.print_exc()
            raise

    def _save_data(self, base_path):
        """Saves distinct files for each stage in shards from temp files."""
        try:
            base_dir = os.path.dirname(base_path)
            if base_dir:
                os.makedirs(base_dir, exist_ok=True)
            logger.info("Saving 3-stage cascade data to disk...")
            max_items_per_shard = 20  # Reduced from 50 to speed up processing
            # Initialize shard indices per stage (graphs/labels must stay lockstep)
            shard_indices = {s: 0 for s in (1, 2, 3)}
            for idx, temp_file in enumerate(self.temp_files):
                print(
                    f"Processing temp file {idx+1}/{len(self.temp_files)}: {temp_file}")
                if os.path.isfile(temp_file):
                    print(f"  Loading temp file {temp_file}")
                    partial = torch.load(
                        temp_file, map_location="cpu", weights_only=False)
                else:
                    # For processed projects, it's (p_name, project_results)
                    p_name, project_results = temp_file
                    partial = {
                        "stage1": {"graphs": [(p_name, project_results["stage1"]["graphs"])], "labels": [(p_name, project_results["stage1"]["labels"])]},
                        "stage2": {"graphs": [(p_name, project_results["stage2"]["graphs"])], "labels": [(p_name, project_results["stage2"]["labels"])]},
                        "stage3": {"graphs": [(p_name, project_results["stage3"]["graphs"])], "labels": [(p_name, project_results["stage3"]["labels"])]},
                    }
                for s in [1, 2, 3]:
                    key = f"stage{s}"
                    data = partial.get(key, {})

                    # Paths
                    # Use preferred naming (still load-compatible with legacy via _find_existing_stage_pair)
                    g_path = self._stage_paths_candidates(
                        base_path, s, "graph")[0]
                    l_path = self._stage_paths_candidates(
                        base_path, s, "label")[0]

                    if "graphs" in data:
                        print(f"  Flattening stage {s} data...")
                        graphs_map = {}
                        labels_map = {}

                        # Store as triples to avoid unsafe parsing/splitting on underscores.
                        # Keyed by (project_name, subkey) to guarantee exact graph/label alignment.
                        for p_name, graph_dict in data.get("graphs", []):
                            if not isinstance(graph_dict, dict):
                                print(
                                    f"Stage {s} graphs for {p_name} is not a dict; skipping")
                                continue
                            for subkey, g in graph_dict.items():
                                graphs_map[(p_name, subkey)] = g

                        for p_name, label_dict in data.get("labels", []):
                            if not isinstance(label_dict, dict):
                                print(
                                    f"Stage {s} labels for {p_name} is not a dict; skipping")
                                continue
                            for subkey, lbl in label_dict.items():
                                labels_map[(p_name, subkey)] = lbl

                        graph_keys = set(graphs_map.keys())
                        label_keys = set(labels_map.keys())
                        if graph_keys != label_keys:
                            missing = sorted(
                                list(graph_keys - label_keys))[:10]
                            extra = sorted(list(label_keys - graph_keys))[:10]
                            raise ValueError(
                                f"Stage {s} graph/label key mismatch while saving: "
                                f"graphs={len(graph_keys)} labels={len(label_keys)} "
                                f"missing_labels={missing} extra_labels={extra}"
                            )

                        # Deterministic ordering for stable shard composition
                        ordered_keys = sorted(
                            graph_keys, key=lambda x: (str(x[0]), str(x[1])))
                        flattened_graphs = [(p, sub, graphs_map[(p, sub)])
                                            for (p, sub) in ordered_keys]
                        flattened_labels = [(p, sub, labels_map[(p, sub)])
                                            for (p, sub) in ordered_keys]

                        print(
                            f"  Flattened {len(flattened_graphs)} graphs and {len(flattened_labels)} labels for stage {s}")

                        print(
                            f"  Saving {len(flattened_graphs)} items in shards for stage {s}...")
                        # Save graphs+labels in lockstep shards to guarantee load completeness.
                        total_saved_graphs = 0
                        total_saved_labels = 0
                        if flattened_graphs:
                            assert len(flattened_graphs) == len(
                                flattened_labels)
                            for start in range(0, len(flattened_graphs), max_items_per_shard):
                                end = min(start + max_items_per_shard,
                                          len(flattened_graphs))
                                g_chunk = flattened_graphs[start:end]
                                l_chunk = flattened_labels[start:end]
                                shard_idx = shard_indices[s]
                                g_shard_path = g_path.replace(
                                    '.pt', f'_{shard_idx}.pt')
                                l_shard_path = l_path.replace(
                                    '.pt', f'_{shard_idx}.pt')
                                torch.save(g_chunk, g_shard_path)
                                torch.save(l_chunk, l_shard_path)
                                total_saved_graphs += len(g_chunk)
                                total_saved_labels += len(l_chunk)
                                shard_indices[s] += 1

                        print(
                            f"  Total saved: {total_saved_graphs} graphs, {total_saved_labels} labels for stage {s}")

                        logger.info(
                            f"  Saved partial Stage {s}: {len(flattened_graphs)} items")
                del partial  # Free memory after processing each partial
                import gc
                gc.collect()
                print(f"  Finished processing temp file {idx+1}")
            # Clean up temp files
            for temp_file in self.temp_files:
                if os.path.isfile(temp_file):
                    os.remove(temp_file)
            print("All temp files processed and cleaned up.")

        except Exception as e:
            logger.error(f"Failed to save data: {e}")

    def _load_data(self, base_path):
        """Loads the current stage file, supporting both old single files and new shards."""
        s = self.stage
        logger.info(f"[YELLOW] Attempting to load Stage {s} data from disk...")

        # Show expected candidate paths before resolving.
        try:
            g_cands = self._stage_paths_candidates(base_path, s, "graph")
            l_cands = self._stage_paths_candidates(base_path, s, "label")
            _print_path_banner(
                f"LOAD (CANDIDATES) STAGE {s}",
                [
                    ("base_path", os.path.abspath(base_path)),
                    ("graph_candidate[0]", os.path.abspath(
                        g_cands[0]) if g_cands else "<none>"),
                    ("label_candidate[0]", os.path.abspath(
                        l_cands[0]) if l_cands else "<none>"),
                ],
            )
        except Exception:
            pass
        pair = self._find_existing_stage_pair(base_path, s)
        if pair is None:
            logger.warning(
                f"No files (single or sharded) found for Stage {s} at base {base_path}"
            )
            return False

        g_path, l_path = pair

        _print_path_banner(
            f"LOAD (RESOLVED) STAGE {s}",
            [
                ("graph_base", os.path.abspath(g_path)),
                ("label_base", os.path.abspath(l_path)),
                ("graph_shards", os.path.abspath(g_path).replace(".pt", "_*.pt")),
                ("label_shards", os.path.abspath(l_path).replace(".pt", "_*.pt")),
            ],
        )

        flattened_graphs = []
        flattened_labels = []

        # Final containers stored into self.dataset_graph/self.dataset_label.
        # For shard mode we fill these directly (streamed), for single-file we deflatten.
        graphs_by_project = {}
        labels_by_project = {}

        is_single = os.path.exists(g_path) and os.path.exists(l_path)
        if is_single:
            # Old way: single files
            logger.info(f"Loading Stage {s} from single files...")
            try:
                flattened_graphs = torch.load(
                    g_path, map_location="cpu", weights_only=False, mmap=True
                )
            except TypeError:
                flattened_graphs = torch.load(g_path, map_location="cpu")

            try:
                flattened_labels = torch.load(
                    l_path, map_location="cpu", weights_only=False, mmap=True
                )
            except TypeError:
                flattened_labels = torch.load(l_path, map_location="cpu")
        else:
            # New sharded way
            logger.info(f"Loading Stage {s} from shards...")
            shard_paths = []
            shard_idx = 0
            while True:
                g_shard_path = g_path.replace('.pt', f'_{shard_idx}.pt')
                l_shard_path = l_path.replace('.pt', f'_{shard_idx}.pt')
                if not (os.path.exists(g_shard_path) and os.path.exists(l_shard_path)):
                    break
                shard_paths.append((g_shard_path, l_shard_path))
                shard_idx += 1
            if not shard_paths:
                logger.warning(
                    f"No shards found for Stage {s} at {g_path}")
                return False

            # Parallel load shards
            def load_shard(g_path, l_path):
                try:
                    chunk_g = torch.load(
                        g_path, map_location="cpu", weights_only=False, mmap=True)
                except TypeError:
                    chunk_g = torch.load(g_path, map_location="cpu")

                try:
                    chunk_l = torch.load(
                        l_path, map_location="cpu", weights_only=False, mmap=True)
                except TypeError:
                    chunk_l = torch.load(l_path, map_location="cpu")
                return chunk_g, chunk_l

            with ThreadPoolExecutor(max_workers=min(len(shard_paths), 4)) as executor:
                futures = [executor.submit(load_shard, g_p, l_p) for g_p, l_p in shard_paths]
                for future in as_completed(futures):
                    chunk_g, chunk_l = future.result()
                    # Stream-deflatten to reduce peak memory.
                    if not isinstance(chunk_g, list) or not isinstance(chunk_l, list):
                        raise TypeError(
                            f"Stage {s} shard: expected list chunks, got graphs={type(chunk_g)} labels={type(chunk_l)}"
                        )
                    if len(chunk_g) != len(chunk_l):
                        raise ValueError(
                            f"Stage {s} shard: chunk size mismatch graphs={len(chunk_g)} labels={len(chunk_l)}"
                        )

                    for gi, li in zip(chunk_g, chunk_l):
                        if not (isinstance(gi, (list, tuple)) and len(gi) == 3):
                            raise TypeError(
                                f"Stage {s} shard: bad graph item type={type(gi)}"
                            )
                        if not (isinstance(li, (list, tuple)) and len(li) == 3):
                            raise TypeError(
                                f"Stage {s} shard: bad label item type={type(li)}"
                            )
                        gp, gsub, gobj = gi
                        lp, lsub, lobj = li
                        if gp != lp or gsub != lsub:
                            raise ValueError(
                                f"Stage {s} shard: graph/label misaligned: "
                                f"graph=({gp},{gsub}) label=({lp},{lsub})"
                            )
                        graphs_by_project.setdefault(gp, {})[gsub] = gobj
                        labels_by_project.setdefault(lp, {})[lsub] = lobj

                    # Free shard chunks promptly
                    del chunk_g, chunk_l
        if is_single:
            print(
                f"Loaded {len(flattened_graphs)} graphs and {len(flattened_labels)} labels for stage {s}")

            if len(flattened_graphs) != len(flattened_labels):
                logger.warning(
                    f"Stage {s} loaded counts differ: graphs={len(flattened_graphs)} labels={len(flattened_labels)}"
                )

            # Deflatten (single-file load only)
            for item in flattened_graphs:
                if isinstance(item, (list, tuple)) and len(item) == 3:
                    p_name, subkey, g = item
                    graphs_by_project.setdefault(p_name, {})[subkey] = g
                else:
                    logger.warning(
                        f"Unexpected graph item format: {type(item)}")

            for item in flattened_labels:
                if isinstance(item, (list, tuple)) and len(item) == 3:
                    p_name, subkey, lbl = item
                    labels_by_project.setdefault(p_name, {})[subkey] = lbl
                else:
                    logger.warning(
                        f"Unexpected label item format: {type(item)}")
        else:
            total_graph_items = sum(len(gdict)
                                    for gdict in graphs_by_project.values())
            total_label_items = sum(len(ldict)
                                    for ldict in labels_by_project.values())
            print(
                f"Loaded {total_graph_items} graphs and {total_label_items} labels for stage {s} (parallel shards)")

        total_graph_items = sum(len(gdict)
                                for gdict in graphs_by_project.values())
        total_label_items = sum(len(ldict)
                                for ldict in labels_by_project.values())
        print(
            f"Deflattened: {total_graph_items} graph items, {total_label_items} label items")

        if total_graph_items != total_label_items:
            raise ValueError(
                f"Stage {s} deflatten mismatch: graphs={total_graph_items} labels={total_label_items}"
            )
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

        # Free memory
        del flattened_graphs, flattened_labels
        import gc
        gc.collect()
        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass

        return True

    def _process_single_item(self, p_name_g, subkey, graph_data, lbl):
        full_key = f"{p_name_g}@{subkey}"

        if self.stage in [1, 2]:
            g = graph_data

            if not hasattr(g, "ntypes"):
                raise TypeError(
                    f"Stage 1/2 expects a DGL heterograph, got {type(g)} for {full_key}")

            if not isinstance(lbl, int) or lbl not in (0, 1):
                raise ValueError(
                    f"Stage {self.stage} label must be int 0/1 for {full_key}, got {lbl} ({type(lbl)})"
                )

            g = standardize_heterograph(
                g,
                getattr(self.CPG_Proccessor, "rel_names", None),
                self.embedding_dims,
            )

            graph_tuple = (full_key, g)
            label_tuple = (full_key, torch.tensor(lbl, dtype=torch.float32))

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

            expected_dim = len(
                getattr(_graph_utils, "OWASP_VULN", []))

            cfg_nodes = g.num_nodes(
                "cfg_node") if "cfg_node" in g.ntypes else 0
            ast_nodes = g.num_nodes(
                "ast_node") if "ast_node" in g.ntypes else 0

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
                return None

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

            graph_tuple = (full_key, g)
            label_tuple = (
                full_key,
                {
                    "cfg_node": cfg_tensor,
                    "ast_node": ast_tensor,
                },
            )

        return graph_tuple, label_tuple

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

        # Collect rel_names from graphs
        all_rels = set()
        for p_name_g, graphs_data in graphs_by_project.items():
            for subkey, graph_data in graphs_data.items():
                if hasattr(graph_data, 'canonical_etypes'):
                    all_rels.update(graph_data.canonical_etypes)
        self.rel_names = list(all_rels)

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
            items_to_process = []
            for p_name_g, graphs_data in graphs_by_project.items():
                labels_data = labels_by_project.get(p_name_g, {})

                if not isinstance(graphs_data, dict) or not isinstance(labels_data, dict):
                    logger.warning(
                        f"Unexpected per-project data types for {p_name_g}: graphs={type(graphs_data)} labels={type(labels_data)}"
                    )
                    continue

                # Collect items for parallel processing
                for subkey, graph_data in graphs_data.items():
                    lbl = labels_data.get(subkey)
                    if lbl is None:
                        raise ValueError(
                            f"Missing label for {p_name_g}@{subkey}. This indicates stage graph/label keys are inconsistent."
                        )
                    items_to_process.append((p_name_g, subkey, graph_data, lbl))

            # Process items in parallel
            import concurrent.futures
            max_workers = min(4, len(items_to_process)) if len(items_to_process) > 0 else 1
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                results = list(executor.map(lambda item: self._process_single_item(*item), items_to_process))

            # Collect results
            for result in results:
                if result is not None:
                    graph_tuple, label_tuple = result
                    dataset_graph_temp.append(graph_tuple)
                    dataset_label_temp.append(label_tuple)
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

        # Free memory
        del dataset_graph_temp, dataset_label_temp
        import gc
        gc.collect()

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
        choices=["DAppSCAN", "MANDO", "EtherScanIO"],
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
    _ = CustomDataset(source=args.source,
                      stage=3, force_reload=args.force_reload, is_test=args.test)
