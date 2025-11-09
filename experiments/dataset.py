"""
Dataset classes for ChainGuard.
Handles data loading and preprocessing for training, validation, and testing.
"""

import sys
import os
import traceback

import dgl
# Add the workspace root to Python path
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

import json
import random
import numpy as np
import torch
import pandas as pd
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer
from torch.utils.data import Dataset
import torch.nn.utils.rnn as rnn_utils

from Data.DAppSCAN.f6_DAppSCAN_fetch_data_adapter import f6_fetch_DAppSCAN_data

from experiments.graph_processing import GraphFeatureExtractor
from experiments.raw_code_processing import RawCodeFeatureExtractor
from experiments.the_utils.logger import setup_logger
from datetime import datetime

# We'll set LOG_FOLDER from environment or use default
log_folder = os.getenv('LOG_FOLDER', 'Logs')
os.makedirs(log_folder, exist_ok=True)
logger = setup_logger(f"{log_folder}/Dataset.log")

os.environ['DGLBACKEND'] = 'pytorch'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
## DEFINE INPUTS

class CustomDataset(Dataset):
    """Dataset class for loading and processing data."""

    def __init__(self, tokenizer, args, source = "DAppSCAN", file_path = "./train.txt" , force_reload=False, load_type="both", rand_seed=42):
        # region INIT
        logger.info("Initializing CustomDataset...")
        self.set_rand_seed(rand_seed)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.warning(f"Using device: {self.device}")
        
        self.args = args
        self.dataset_code = []
        self.dataset_graph = []

        if force_reload:
            self.tokenizer = AutoTokenizer.from_pretrained('microsoft/codebert-base', use_fast=True)
            self.embedding_model = AutoModel.from_pretrained('microsoft/codebert-base')
        else:
            self.tokenizer = None
            self.embedding_model = None
            logger.warning("Only load data mode !")
            
        self.GraphFeatureExtractorModule = GraphFeatureExtractor(None, 
                                                                self.device)
        
        self.RawCodeFeatureExtractorModule = RawCodeFeatureExtractor(self.tokenizer, 
                                                                    self.embedding_model, 
                                                                    self.device)
        
        self.ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.save_load_dir = "./processed_data"

        self.data_feat_dims = {}
        self.fetch_data_feat_dims()
        os.makedirs(self.save_load_dir, exist_ok=True)
        if source == "DAppSCAN":
            self._fetch_and_process_DAppSCAN_data(force_reload=force_reload, load_type=load_type)
        
    def set_rand_seed(self, rand_seed):
        random.seed(rand_seed)
        np.random.seed(rand_seed)
        torch.manual_seed(rand_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(rand_seed)

    def fetch_data_feat_dims(self):
        graph_dims = self.GraphFeatureExtractorModule.fetch_all_feat_dim()
        code_dims = self.RawCodeFeatureExtractorModule.fetch_all_feat_dim()
        self.data_feat_dims.update(graph_dims)
        self.data_feat_dims.update(code_dims)
        logger.info(f"Fetched feature dimensions: \n{json.dumps(self.data_feat_dims,indent=2)}")


    def _fetch_and_process_DAppSCAN_data(self, force_reload=False, load_type="both"):
        # region DAppSCAN
        saved_file = os.path.join(self.save_load_dir, f"DAppSCAN_dataset.pt")
        try:
            #check if data already processed
            if not force_reload:
                x = self._load_data(saved_file, type="both")
                if x:
                    logger.warning(f"DAppSCAN data loaded from previous.")
                    return
                else:
                    logger.warning(f"No existing DAppSCAN data found. Proceeding to fetch and process.")
            else:
                logger.warning(f"Forcing reload and preprocessing of DAppSCAN data {load_type}.")
                if load_type in ["graph","code"]:
                    self._load_data(saved_file, load_type)
                    logger.warning(f"Loaded existing {load_type} data for reprocessing.")            
                else:
                    logger.warning(f"Full reload initiated for DAppSCAN data.")
                    # prompt user to confirm
                    logger.critical(f"This will reset all saves in {saved_file.replace('_dataset.pt', '_code_dataset.pt')} and {saved_file.replace('_dataset.pt', '_graph_dataset.pt')}")
                    input("Press Enter to continue... , to cancel press Ctrl+C")

            logger.info("Fetching and processing DAppSCAN data...")
            res = f6_fetch_DAppSCAN_data(root=r"Data/DAppSCAN/Processed_Data") 
            if not res:
                logger.error("No data fetched from DAppSCAN. Please check the data source.")
                sys.exit(1)
            graph_list, df_list = res
            print(f"Fetched: {len(graph_list)} graphs - {len(df_list)} code")
            if not graph_list:
                logger.error("No graphs fetched from DAppSCAN. Please check the data source.")
                sys.exit(1)
            if not df_list:
                logger.error("No source code data fetched from DAppSCAN. Please check the data source.")
                sys.exit(1)
                
            source_code_list = []
            hetero_cpg_graph_list = []
            if load_type != "graph":
                hetero_cpg_graph_list =  self.GraphFeatureExtractorModule.pipeline_graphs_to_features(graph_list)

            if load_type != "code":
                source_code_list = self.RawCodeFeatureExtractorModule.pipeline_process_code_csv_to_features(df_list)

            if not hetero_cpg_graph_list and not self.dataset_graph:
                print(hetero_cpg_graph_list)
                print(self.dataset_graph)
                logger.error("[GRAPH] Processing resulted in empty data. Please check the processing steps.")
                return

            if not source_code_list and not self.dataset_code:
                print(source_code_list)
                print(self.dataset_code)
                logger.error("[CODE] Processing resulted in empty data. Please check the processing steps.")
                return

            if max(len(hetero_cpg_graph_list), len(self.dataset_graph)) != max(len(source_code_list), len(self.dataset_code)):
                logger.error(f"Mismatch in number of processed graphs {len(hetero_cpg_graph_list), len(self.dataset_graph)} and source codes {len(source_code_list), len(self.dataset_code)}.")
                return
            
            if max(len(hetero_cpg_graph_list), len(self.dataset_graph)) > 0:
                if (len(hetero_cpg_graph_list) > 0):
                    self.dataset_graph = hetero_cpg_graph_list

                self._save_data(saved_file, type="graph")
                logger.info(f"Saved processed graph data to {saved_file.replace('_dataset.pt', '_graph_dataset.pt')}")

            if max(len(source_code_list), len(self.dataset_code)) >= 0:
                if (len(source_code_list) > 0):
                    self.dataset_code = source_code_list
                self._save_data(saved_file, type="code")
                logger.info(f"[SUCCESS] Saved processed code data to {saved_file.replace('_dataset.pt', '_code_dataset.pt')}")
            
            # move to cpu for next steps
            # Each item is a tuple: (project_name, data_dict)
            # We need to move the tensors inside data_dict to CPU
            self.dataset_graph = [
                (name, {k: v.to("cpu") if isinstance(v, torch.Tensor) or hasattr(v, 'to') else v 
                        for k, v in data.items()}) 
                for name, data in self.dataset_graph
            ]
            self.dataset_code = [
                (name, {k: v.to("cpu") if isinstance(v, torch.Tensor) else v 
                        for k, v in data.items()}) 
                for name, data in self.dataset_code
            ]
            
            logger.info(f"Saved interactive graph visualization to graph_visualization.html")
        except Exception as e:
            traceback.print_exc()

    #####################################################################
    def _save_data(self, save_path , type="both"):
        try:
            os.makedirs(self.save_load_dir, exist_ok=True)
            # region SAVE DATA
            if type == "both":
                torch.save(self.dataset_code, save_path.replace("_dataset.pt", "_code_dataset.pt"))
                torch.save(self.dataset_graph, save_path.replace("_dataset.pt", "_graph_dataset.pt"))
                logger.info(f"Data saved to {save_path.replace('_dataset.pt', '_code_dataset.pt')} and {save_path.replace('_dataset.pt', '_graph_dataset.pt')}")
            elif type == "code":
                torch.save(self.dataset_code, save_path.replace("_dataset.pt", "_code_dataset.pt"))
                logger.info(f"Data saved to {save_path.replace('_dataset.pt', '_code_dataset.pt')}")
            elif type == "graph":
                torch.save(self.dataset_graph, save_path.replace("_dataset.pt", "_graph_dataset.pt"))
                logger.info(f"Data saved to {save_path.replace('_dataset.pt', '_graph_dataset.pt')}")
        except Exception as e:
            logger.error(f"Failed to save data to {save_path}: {e}")
            traceback.print_exc()    
    #####################################################################
    def _load_data(self, load_path, type="both"):
        try:
            os.makedirs(self.save_load_dir, exist_ok=True)
            # region LOAD DATA
            if type == "both":
                code_path = load_path.replace("_dataset.pt", "_code_dataset.pt")
                graph_path = load_path.replace("_dataset.pt", "_graph_dataset.pt")
                if not os.path.exists(code_path) or not os.path.exists(graph_path):
                    logger.error(f"Load paths {code_path} or {graph_path} do not exist.")
                    return None
                self.dataset_code = torch.load(code_path, weights_only=False, map_location='cpu')
                self.dataset_graph = torch.load(graph_path, weights_only=False, map_location='cpu')
                return True
            elif type == "code":
                code_path = load_path.replace("_dataset.pt", "_code_dataset.pt")
                if not os.path.exists(code_path):
                    logger.error(f"Load path {code_path} does not exist.")
                    return None
                self.dataset_code = torch.load(code_path, weights_only=False, map_location='cpu')
                return True
            elif type == "graph":
                graph_path = load_path.replace("_dataset.pt", "_graph_dataset.pt")
                if not os.path.exists(graph_path):
                    logger.error(f"Load path {graph_path} does not exist.")
                    return None
                self.dataset_graph = torch.load(graph_path, weights_only=False, map_location='cpu')
                return True
        except Exception as e:
            logger.error(f"Failed to load data from {load_path}: {e}")
            traceback.print_exc()
    #####################################################################

    def __len__(self):
        code = len(self.dataset_code) if self.dataset_code else 0
        graph = len(self.dataset_graph) if self.dataset_graph else 0
        if code != graph:
            logger.warning(f"Length mismatch between code ({code}) and graph ({graph}) datasets.")
        return code

    #####################################################################
    def __getitem__(self, item):
        # item = index eg. 0,1,2,... = a[1] 
        try:
            # Get graph data
            graph_features = None
            graph_cg_labels = None
            graph_cfg_labels = None
            graph_dfg_labels = None
            g_project_name = None
            if self.dataset_graph and item < len(self.dataset_graph):
                graph_item = self.dataset_graph[item]
                if isinstance(graph_item, tuple):
                    g_project_name , graph_data = graph_item
                    
                    if isinstance(graph_data, dict):
                        graph_features = graph_data.get('graph', None)
                        graph_cg_labels = graph_data.get('cg_vuln', None)
                        graph_cfg_labels = graph_data.get('cfg_vuln', None)
                        graph_dfg_labels = graph_data.get('dfg_vuln', None)

                    if not graph_features:
                        logger.warning(f"Graph features missing for item {item}")
            else:
                logger.warning(f"Graph data not available for item {item}")
            
            # Get code data
            code_features = None
            code_labels = None
            c_project_name = None
            if self.dataset_code and item < len(self.dataset_code):
                code_item = self.dataset_code[item]
                if isinstance(code_item, tuple):
                    c_project_name , code_data = code_item

                    ######### ENSURE PROJECT NAMES MATCH #########
                    if g_project_name  and c_project_name and g_project_name != c_project_name:
                        logger.critical(f"Project name mismatch between graph ({g_project_name}) and code ({c_project_name}) at item {item}")
                        sys.exit(1)

                    if isinstance(code_data, dict):
                        code_features = code_data.get('features', None) 
                        code_labels = code_data.get('labels', None)
                        
                        # DEEP VALIDATION: Check if labels might have leaked into features
                        if code_features is not None and code_labels is not None:
                            # Check if feature dimension matches label dimension (potential leak)
                            if code_features.shape[-1] == code_labels.shape[-1]:
                                logger.warning(f"Sample {item}: Feature dim ({code_features.shape[-1]}) == Label dim ({code_labels.shape[-1]}) - potential leak?")
                            
                            # Check if features and labels are suspiciously similar
                            if code_features.shape == code_labels.shape:
                                similarity = torch.nn.functional.cosine_similarity(
                                    code_features.flatten(), 
                                    code_labels.flatten(), 
                                    dim=0
                                ).item()
                                if similarity > 0.9:
                                    logger.error(f"Sample {item}: Features and labels are {similarity:.4f} similar - CRITICAL LEAK!")
                
                # Check if tensor is valid (not None and has data)
                if code_features is None or (isinstance(code_features, torch.Tensor) and code_features.numel() == 0):
                    logger.warning(f"Code features missing for item {item}")
            else:
                logger.warning(f"Code data not available for item {item}")
            
            # ADDITIONAL VALIDATION: Check graph-label alignment
            if graph_features is not None and graph_cg_labels is not None:
                num_functions = graph_features.num_nodes('function') if hasattr(graph_features, 'num_nodes') else 0
                num_cg_labels = graph_cg_labels.shape[0] if isinstance(graph_cg_labels, torch.Tensor) else 0
                
                if num_functions != num_cg_labels:
                    logger.warning(f"Sample {item}: Graph has {num_functions} functions but {num_cg_labels} CG labels")
            
            if graph_features is not None and graph_cfg_labels is not None:
                num_blocks = graph_features.num_nodes('block') if hasattr(graph_features, 'num_nodes') else 0
                num_cfg_labels = graph_cfg_labels.shape[0] if isinstance(graph_cfg_labels, torch.Tensor) else 0
                
                if num_blocks != num_cfg_labels:
                    logger.warning(f"Sample {item}: Graph has {num_blocks} blocks but {num_cfg_labels} CFG labels")
            
            if graph_features is not None and graph_dfg_labels is not None:
                num_dfg_nodes = graph_features.num_nodes('dfg_node') if hasattr(graph_features, 'num_nodes') else 0
                num_dfg_labels = graph_dfg_labels.shape[0] if isinstance(graph_dfg_labels, torch.Tensor) else 0
                
                if num_dfg_nodes != num_dfg_labels:
                    logger.warning(f"Sample {item}: Graph has {num_dfg_nodes} DFG nodes but {num_dfg_labels} DFG labels")
            
            return {
                'graph': graph_features,
                'cg_labels': graph_cg_labels,
                'cfg_labels': graph_cfg_labels,
                'dfg_labels': graph_dfg_labels,
                'code': code_features,
                'labels': code_labels,
            }
        
        except Exception as e:
            logger.error(f"Error retrieving item {item}: {e}")
            traceback.print_exc()
            # Return empty data on error
            return {
                'graph': None,
                'cg_labels': None,
                'cfg_labels': None,
                'dfg_labels': None,
                'code': None,
                'labels': None,
            }

# region collatte function for DataLoader
def custom_collate(batch):
    # 'batch' is a list of dictionaries, where each dict is 
    # the output of your __getitem__

    # --- 2. Handle DGL Graphs ---
    # Use dgl.batch to combine all graphs into one large graph
    graphs = [item['graph'] for item in batch]
    batched_graph = dgl.batch(graphs)
    
    # --- 3. Handle Graph-Level Labels ---
    # cg_labels = torch.stack([item['cg_labels'] for item in batch])
    # cfg_labels = torch.stack([item['cfg_labels'] for item in batch])
    cg_labels_list = [item['cg_labels'] for item in batch]
    cfg_labels_list = [item['cfg_labels'] for item in batch]
    dfg_labels_list = [item['dfg_labels'] for item in batch]
    # Get lengths for all of them
        
    # Get lengths for all of them (handle None values)
    cg_lengths = torch.tensor([len(c) if c is not None else 0 for c in cg_labels_list])
    cfg_lengths = torch.tensor([len(c) if c is not None else 0 for c in cfg_labels_list])
    dfg_lengths = torch.tensor([len(d) if d is not None else 0 for d in dfg_labels_list])
    
    # Filter out None values before padding - replace with empty tensors
    cg_labels_filtered = [c if c is not None else torch.empty(0, dtype=torch.float32) for c in cg_labels_list]
    cfg_labels_filtered = [c if c is not None else torch.empty(0, dtype=torch.float32) for c in cfg_labels_list]
    dfg_labels_filtered = [d if d is not None else torch.empty(0, dtype=torch.float32) for d in dfg_labels_list]

    padded_cg_labels = rnn_utils.pad_sequence(
            cg_labels_filtered, 
            batch_first=True, 
            padding_value=-1
        )
    padded_cfg_labels = rnn_utils.pad_sequence(
        cfg_labels_filtered, 
        batch_first=True, 
        padding_value=-1
    )
    padded_dfg_labels = rnn_utils.pad_sequence(
        dfg_labels_filtered, 
        batch_first=True, 
        padding_value=-1
    )
    # --- 4. Handle Variable-Length Code/Labels (The tricky part) ---
    code_features_list = [item['code'] for item in batch]
    line_labels_list = [item['labels'] for item in batch]
    
    # Get the original lengths BEFORE padding. This is crucial for your model.
    code_lengths = torch.tensor([len(c) for c in code_features_list])
    
    # Pad the sequences
    # batch_first=True makes the output shape:
    # (batch_size, max_sequence_length, feature_dim)
    padded_code = rnn_utils.pad_sequence(
        code_features_list, 
        batch_first=True, 
        padding_value=0.0
    )
    
    # padding_value=-1 (or ignore_index) is common for labels
    padded_labels = rnn_utils.pad_sequence(
        line_labels_list, 
        batch_first=True, 
        padding_value=-1 
    )
    logger.info("="*40)
    logger.info(f"Batched graph has {batched_graph.num_nodes()} nodes and {batched_graph.num_edges()} edges.")
    logger.info(f"CG labels shape: {padded_cg_labels.shape}")
    logger.info(f"CFG labels shape: {padded_cfg_labels.shape}")
    logger.info(f"DFG labels shape: {padded_dfg_labels.shape}")
    logger.info(f"Padded code shape: {padded_code.shape}")
    logger.info(f"Padded labels shape: {padded_labels.shape}")
    logger.info(f"Code lengths: {code_lengths}")
    logger.info("="*40)
    # --- 5. Return the Batched Dictionary ---
    return {
            'graph': batched_graph,       
            'code': padded_code,          # Padded tensor
            'labels': padded_labels,      # Padded tensor
            'cg_labels': padded_cg_labels,   # Padded tensor
            'cfg_labels': padded_cfg_labels, # Padded tensor
            'dfg_labels': padded_dfg_labels, # Padded tensor
            'code_lengths': code_lengths, # Lengths tensor
            'cg_lengths': cg_lengths,     # Lengths tensor
            'cfg_lengths': cfg_lengths,   # Lengths tensor
            'dfg_lengths': dfg_lengths,   # Lengths tensor
        }

if __name__ == "__main__":
    # region TESTING
    import argparse
    
    parser = argparse.ArgumentParser(description='Test dataset loading and inspection')
    parser.add_argument('--force_reload', action='store_true', 
                        help='Force reload and reprocess data from source')
    parser.add_argument('--load_type', type=str, default='both', 
                        choices=['both', 'graph', 'code'],
                        help='Type of data to load: both, graph, or code')
    args = parser.parse_args()
    
    dataset = CustomDataset(
        "cccc", 
        None, 
        source="DAppSCAN", 
        file_path="./train.txt",
        force_reload=args.force_reload, 
        load_type=args.load_type
    )
    # Print dataset length
    print(f"Dataset loaded with {len(dataset.dataset_graph)} - {len(dataset.dataset_code)} samples.")
    # Print a sample item and details
    for i in range(min(2, len(dataset))):
        item = dataset[i]
        print(f"Sample {i}:")
        g = item['graph']
        print(f"Graph object: {g}")
        if g is not None:
            try:
                print(f"Total nodes (all types): {g.num_nodes()}")
            except Exception:
                print("Total nodes: (could not retrieve)")
            try:
                print(f"Total edges (all types): {g.num_edges()}")
            except Exception:
                print("Total edges: (could not retrieve)")

            # Per-node-type information
            print("Node types and feature shapes:")
            try:
                for ntype in g.ntypes:
                    try:
                        ncount = g.num_nodes(ntype)
                    except Exception:
                        ncount = 'unknown'
                    ndata_keys = list(g.nodes[ntype].data.keys()) if hasattr(g, 'nodes') else []
                    feat = None
                    if 'feat' in ndata_keys:
                        feat = g.nodes[ntype].data.get('feat', None)
                    # Print shape if available
                    if isinstance(feat, torch.Tensor):
                        print(f"  - {ntype}: count={ncount}, feat_shape={tuple(feat.shape)} (ndata keys: {ndata_keys})")
                    else:
                        print(f"  - {ntype}: count={ncount}, no 'feat' tensor found (ndata keys: {ndata_keys})")
            except Exception as e:
                print(f"  [Error reading node types]: {e}")

            # Per-edge-type information (canonical etypes)
            print("Edge (canonical) types and feature shapes:")
            try:
                for etype in g.canonical_etypes:
                    # etype is a tuple (src, rel, dst)
                    try:
                        ecount = g.num_edges(etype)
                    except Exception:
                        ecount = 'unknown'
                    # Access edge data for this relation
                    edata_keys = []
                    try:
                        edata_keys = list(g.edges[etype].data.keys())
                    except Exception:
                        # Some DGL versions use g.edge_type_subgraph etc.; fallback
                        try:
                            edata_keys = list(g.edata.keys())
                        except Exception:
                            edata_keys = []

                    efeat = None
                    try:
                        # Preferred access for heterograph: g.edges[etype].data.get('feat')
                        efeat = g.edges[etype].data.get('feat', None)
                    except Exception:
                        try:
                            # Fallback: check global edata
                            efeat = g.edata.get('feat', None)
                        except Exception:
                            efeat = None

                    if isinstance(efeat, torch.Tensor):
                        print(f"  - {etype}: count={ecount}, feat_shape={tuple(efeat.shape)} (edata keys: {edata_keys})")
                    else:
                        print(f"  - {etype}: count={ecount}, no 'feat' tensor found (edata keys: {edata_keys})")
            except Exception as e:
                print(f"  [Error reading edge types]: {e}")

            # Also list flat edge type strings (some DGL versions or exports store them here)
            print("Flat edge types (g.etypes) and feature shapes (including 'link' edges):")
            try:
                for etype_str in g.etypes:
                    try:
                        ecount = g.num_edges(etype_str)
                    except Exception:
                        ecount = 'unknown'
                    edata_keys = []
                    efeat = None
                    try:
                        # g.edges(etype_str).data is supported for heterographs
                        edata_keys = list(g.edges[etype_str].data.keys())
                        efeat = g.edges[etype_str].data.get('feat', None)
                    except Exception:
                        # Fallback: check global edata or per-relation storage
                        try:
                            edata_keys = list(g.edata.keys())
                            efeat = g.edata.get('feat', None)
                        except Exception:
                            edata_keys = []
                            efeat = None

                    tag = ""
                    if 'link' in str(etype_str).lower():
                        tag = " [LINK EDGE]"

                    if isinstance(efeat, torch.Tensor):
                        print(f"  - {etype_str}: count={ecount}, feat_shape={tuple(efeat.shape)} (edata keys: {edata_keys}){tag}")
                    else:
                        print(f"  - {etype_str}: count={ecount}, no 'feat' tensor found (edata keys: {edata_keys}){tag}")
            except Exception as e:
                print(f"  [Error reading flat edge types]: {e}")

        else:
            print("No graph available for this sample.")

        print(f"CG Labels: {item['cg_labels']}, {item['cg_labels'].shape if item['cg_labels'] is not None else None}")
        print(f"CFG Labels: {item['cfg_labels']}, {item['cfg_labels'].shape if item['cfg_labels'] is not None else None}")
        print(f"DFG Labels: {item['dfg_labels']}, {item['dfg_labels'].shape if item['dfg_labels'] is not None else None}")
        print(f"Code: {item['code']}, {item['code'].shape if item['code'] is not None else None}")
        print(f"Labels: {item['labels']}, {item['labels'].shape if item['labels'] is not None else None}")

