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


def standardize_stage3_heterograph(g, required_rel_names, embedding_dims):
    """Produce a standardized heterograph using only canonical node types.

    Behavior summary:
    - Ensure presence of cfg_node and ast_node (0 nodes if missing).
    - Add all requested canonical edge types (0 edges if missing).
    - Populate node/edge 'feat' with copied tensors when available, otherwise zeros.
    """
    if g is None or not hasattr(g, "ntypes") or not hasattr(g, "canonical_etypes"):
        return g

    # Normalize relation tuples to simple string triples
    def _to_rel(r):
        r = _rel_to_str_tuple(r)
        return r if (isinstance(r, tuple) and len(r) == 3) else None

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

    required_ntypes = set(["cfg_node", "ast_node"])
    for (s, _, d) in req_rels:
        required_ntypes.add(s)
        required_ntypes.add(d)

    # Node counts (0 if missing)
    num_nodes_dict = {nt: int(g.num_nodes(
        nt)) if nt in g.ntypes else 0 for nt in sorted(required_ntypes)}

    # Build edge data mapping for heterograph creation
    data_dict = {}
    existing = set(getattr(g, "canonical_etypes", []))
    for rel in req_rels:
        s, e, d = rel
        if s not in num_nodes_dict or d not in num_nodes_dict:
            continue
        if rel in existing:
            try:
                src, dst = g.edges(etype=rel)
            except Exception:
                src = torch.empty((0,), dtype=torch.int64)
                dst = torch.empty((0,), dtype=torch.int64)
        else:
            src = torch.empty((0,), dtype=torch.int64)
            dst = torch.empty((0,), dtype=torch.int64)
        data_dict[rel] = (src, dst)

    # Ensure at least one relation exists for DGL heterograph creation
    if not data_dict:
        dummy_rel = ("cfg_node", "_dummy", "cfg_node")
        data_dict[dummy_rel] = (torch.empty(
            (0,), dtype=torch.int64), torch.empty((0,), dtype=torch.int64))

    new_g = dgl.heterograph(data_dict, num_nodes_dict=num_nodes_dict)

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
            dim = cfg_dim if ntype == "cfg_node" else (
                ast_dim if ntype == "ast_node" else 0)
            if dim > 0:
                new_g.nodes[ntype].data["feat"] = torch.zeros(
                    (n, dim), dtype=torch.float32)

    # Edge features: copy when possible, otherwise zeros
    for rel in new_g.canonical_etypes:
        m = int(new_g.num_edges(rel))
        if m == 0:
            continue
        feat = None
        if rel in existing:
            try:
                feat = g.edges[rel].data.get("feat")
            except Exception:
                feat = None
        if isinstance(feat, torch.Tensor) and feat.shape[0] == m:
            new_g.edges[rel].data["feat"] = feat.to(torch.float32)
        else:
            if edge_dim > 0:
                new_g.edges[rel].data["feat"] = torch.zeros(
                    (m, edge_dim), dtype=torch.float32)

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
            self.postprocessing()

    def _stage_file_paths(self, stage: int):
        """Returns (graph_path, label_path, embedding_path)."""
        base_filename = "DAppSCAN_dataset.pt"
        base_path = os.path.join(self.save_load_dir, base_filename)
        test_suffix = "test" if self.is_test else ""
        g_path = base_path.replace(
            ".pt", f"_stage{stage}{test_suffix}_graph.pt")
        l_path = base_path.replace(
            ".pt", f"_stage{stage}{test_suffix}_label.pt")

    
        return g_path, l_path

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
            os.makedirs(os.path.dirname(base_path), exist_ok=True)
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
        g_path, l_path = self._stage_file_paths(s)

        if os.path.exists(g_path) and os.path.exists(l_path):
            logger.info(f"Loading Stage {s} from disk...")
            flattened_graphs = torch.load(
                g_path, map_location="cpu", weights_only=False
            )
            flattened_labels = torch.load(
                l_path, map_location="cpu", weights_only=False
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
                            g = standardize_stage3_heterograph(
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
                        g = standardize_stage3_heterograph(
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

        flattened_graphs = torch.load(
            g_path, map_location="cpu", weights_only=False)
        flattened_labels = torch.load(
            l_path, map_location="cpu", weights_only=False)

        graphs_by_project = {}
        for item in flattened_graphs:
            if isinstance(item, (list, tuple)) and len(item) == 3:
                p_name, subkey, g = item
            else:
                raise ValueError(
                    f"Unexpected graph item format in {g_path}: {type(item)} len={len(item) if isinstance(item, (list, tuple)) else 'n/a'}"
                )
            graphs_by_project.setdefault(p_name, {})[subkey] = g

        labels_by_project = {}
        for item in flattened_labels:
            if isinstance(item, (list, tuple)) and len(item) == 3:
                p_name, subkey, lbl = item
            else:
                raise ValueError(
                    f"Unexpected label item format in {l_path}: {type(item)} len={len(item) if isinstance(item, (list, tuple)) else 'n/a'}"
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
        print(
            f"  stage{stage}: graphs={len(projects)}, labels={len(labels_by_project)}")
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
            raise AssertionError(
                f"Stage {stage} produced 0 samples after postprocessing")

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
                    raise RuntimeError(
                        "DGL is required to validate stage 3 graphs")
                g = item["graph"]
                cfg = item["cfg_labels"]
                ast = item["ast_labels"]

                def _pick_ntype(graph, candidates):
                    for cand in candidates:
                        if cand in graph.ntypes:
                            return cand
                    return None

                cfg_ntype = _pick_ntype(g, ["cfg_node"])
                ast_ntype = _pick_ntype(g, ["ast_node"])
                cfg_nodes = g.num_nodes(
                    cfg_ntype) if cfg_ntype is not None else 0
                ast_nodes = g.num_nodes(
                    ast_ntype) if ast_ntype is not None else 0
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
    _ = CustomDataset(
        stage=3, force_reload=args.force_reload, is_test=args.test)

    # Print and assert raw saved schema for all 3 stages.
    for s in [1, 2, 3]:
        _print_and_assert_raw_schema(s)

    # Load each stage, run postprocessing, validate strict alignment.
    for s in [1, 2, 3]:
        ds = CustomDataset(stage=s, force_reload=False, is_test=args.test)
        _validate_postprocessed(ds, s)

        # Print actual sample data (not just schema) for the first few items.
        print(f"\nStage {s} sample data (first 2 items):")
        for i in range(min(30, len(ds))):
            item = ds[i]
            if s in [1, 2]:
                g = item["graph"]
                # Embedding is a real tensor from GraphEmbedder
                key = next(iter(item["graph_labels"].keys()))
                lbl = item["graph_labels"][key]
                print(f"  [{i}] key={key}")
                
                for ntype in g.ntypes:
                    n = g.num_nodes(ntype)
                    feat = g.nodes[ntype].data.get("feat")
                    feat_shape = tuple(feat.shape) if isinstance(
                        feat, torch.Tensor) else None
                    feat_preview = (
                        feat[0, :8].detach().cpu().tolist()
                        if isinstance(feat, torch.Tensor) and feat.numel() > 0
                        else None
                    )
                    print(
                        f"      {ntype}: nodes={n} feat_shape={feat_shape} feat0[:8]={feat_preview}")
            else:
                g = item["graph"]
                cfg = item["cfg_labels"]
                ast = item["ast_labels"]
                print(
                    f"  [{i}] project={item['project_name']} ntypes={list(g.ntypes)}")
                for ntype in g.ntypes:
                    n = g.num_nodes(ntype)
                    feat = g.nodes[ntype].data.get("feat")
                    feat_shape = tuple(feat.shape) if isinstance(
                        feat, torch.Tensor) else None
                    feat_preview = (
                        feat[0, :8].detach().cpu().tolist()
                        if isinstance(feat, torch.Tensor) and feat.numel() > 0
                        else None
                    )
                    print(
                        f"      {ntype}: nodes={n} feat_shape={feat_shape} feat0[:8]={feat_preview}")

                cfg_preview = cfg[:2].detach().cpu().tolist(
                ) if isinstance(cfg, torch.Tensor) else None
                ast_preview = ast[:2].detach().cpu().tolist(
                ) if isinstance(ast, torch.Tensor) else None
                print(
                    f"      cfg_labels: shape={tuple(cfg.shape)} first2={cfg_preview}")
                print(
                    f"      ast_labels: shape={tuple(ast.shape)} first2={ast_preview}")

    print("\n✅ All 3 stages are schema-consistent and postprocessing-aligned.")
