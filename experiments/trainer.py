from datetime import datetime
import logging
from pathlib import Path
import sys
import os
import random
import traceback

import dgl
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from tqdm import tqdm
from dgl.dataloading import GraphDataLoader
from torch.utils.data import random_split, Subset
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score,
    recall_score,
    precision_score,
    accuracy_score,
    precision_recall_curve,
    confusion_matrix,
    f1_score
)

# =====================================================================
# IMPORTANT: Set LOG_FOLDER environment variable BEFORE importing modules
# that use it, so they all use the same log folder
# =====================================================================
log_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")            
log_folder = Path(f'Logs/Logs_{log_time}')
os.environ['LOG_FOLDER'] = str(log_folder)
os.makedirs(log_folder, exist_ok=True)

try:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    sys.path.append(parent_dir)
    from experiments.dataset import CustomDataset, custom_collate
    from experiments.the_utils.logger import setup_logger
    from experiments.models.baseline3 import CombinedModel 
            
except ImportError:
    from dataset import CustomDataset, custom_collate
    from the_utils.logger import setup_logger
    from experiments.models.baseline3 import CombinedModel 

# Now setup the trainer logger
logger = setup_logger(f"{log_folder}/trainer.log", logging.INFO)


os.environ['DGLBACKEND'] = 'pytorch'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

class FocalLoss(nn.Module):
    """
    Implements the Focal Loss for Multilabel Classification.
    
    This loss is designed for extreme class imbalance in multilabel settings.
    For each class independently:
        FL = -alpha * (1-p)^gamma * log(p)           if y=1
        FL = -(1-alpha) * p^gamma * log(1-p)         if y=0
    
    This down-weights easy examples (high confidence correct predictions)
    and focuses training on hard, misclassified examples.
    """
    def __init__(self, alpha=0.25, gamma=2.0, reduction='none'):
        """
        Args:
            alpha (float): Weight for positive class (0-1).
                            Typical: 0.25 for positive, 0.75 for negative.
                            Use higher alpha (0.5-0.75) for rare positives.
            gamma (float): Focusing parameter (0-5). Higher = more focus on hard examples.
                            Typical: 2.0. Use 3-5 for extreme imbalance.
            reduction (str): 'none', 'mean', or 'sum'.
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        Args:
            inputs (torch.Tensor): Raw logits [B, L, C] or [N, C]
            targets (torch.Tensor): Binary labels [B, L, C] or [N, C]
        
        Returns:
            torch.Tensor: Focal Loss (unreduced if reduction='none')
        """
        # Ensure targets are float
        targets = targets.float()
        
        # Get probabilities (0-1 range)
        p = torch.sigmoid(inputs)
        
        # Compute focal loss components separately for positive/negative cases
        # For y=1: FL = -alpha * (1-p)^gamma * log(p)
        # For y=0: FL = -(1-alpha) * p^gamma * log(1-p)
        
        # Clamp probabilities to avoid log(0)
        p_clamped = torch.clamp(p, min=1e-7, max=1.0 - 1e-7)
        
        # Positive case (y=1)
        pos_loss = -self.alpha * torch.pow(1 - p_clamped, self.gamma) * torch.log(p_clamped)
        
        # Negative case (y=0)
        neg_loss = -(1 - self.alpha) * torch.pow(p_clamped, self.gamma) * torch.log(1 - p_clamped)
        
        # Combine based on target
        focal_loss = targets * pos_loss + (1 - targets) * neg_loss

        # Apply reduction if specified
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            # Default: 'none', as required by the trainer's masking function
            return focal_loss


class Trainer:
    # region Initialization
    def __init__(self,
                batch_size=4, 
                num_epochs=500, 
                learning_rate=5e-4,
                patience=10,
                hidden_dim=128, 
                rand_seed=42
                ):
        """Initialize trainer with hyperparameters and data configuration."""
        self.set_rand_seed(rand_seed)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.patience = patience
        self.dataloader_workers = 6
        self.accumulation_steps = 4
        self.fixed_threshold = 0.50
    
        logger.info("=" * 40)
        logger.info("Starting Vulnerability Detection Model Training")
        logger.info("=" * 40)
        logger.info(f"[PURPLE]Device: {self.device}")
        logger.info(f"[PURPLE]Batch size: {self.batch_size}")
        logger.info(f"[PURPLE]Num epochs: {self.num_epochs}")
        logger.info(f"[PURPLE]Learning rate: {self.learning_rate}")
        logger.info(f"[PURPLE]Patience: {self.patience}")
        logger.info(f"[PURPLE]Accumulation steps: {self.accumulation_steps}")
        logger.info(f"[PURPLE]Fixed threshold: {self.fixed_threshold}")

        self.dataset = None
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        
        # Data feature dimensions: AST, CFG, CG, DFG graphs + CodeBERT embeddings
        self.data_feat_dims = {
            "GRAPH_IN_DIMS": {  # SUBJECT TO CHANGE 
                "ast_node": 55,      # AST node feature dimension
                "block": 33,         # CFG block feature dimension
                "function": 57,      # CG function feature dimension
                "dfg_node": 41       # DFG node feature dimension (node_type + 3 hashes + src_length)
            },
            "EDGE_DIMS":{        # SUBJECT TO CHANGE 
                "ast_edge": 4,       # AST edge feature dimension
                "cfg_edge": 3,       # CFG edge feature dimension
                "cg_edge": 3,        # CG edge feature dimension
                "dfg_edge": 4        # DFG edge feature dimension
            },
            "CODE_IN_DIM": 768,  # SUBJECT TO CHANGE # CodeBERT embedding dimension
            "HIDDEN_DIM": hidden_dim,  # Model hidden layer max dimension -> will variable 
            "OUTPUT_DIM": 8          # Number of vulnerability types (multilabel, excluding Benign)
        }

        self.train_perc = 0.8 # 80% for training
        self.val_perc = 0.1 # 10% for validation
        # 10% for testing is implied (1.0 - 0.8 - 0.1)
        # Calculate the number of samples for train and val
        
        self.load_data(rand_seed=rand_seed)

        self.model = None
        self.optimizer = None
        self.scheduler = None

        self.tune_params = {
            'fused_loss_importance': 1.0, # means normal weight for fused loss
            'cg_loss_importance': 1.0, # means normal weight fo CG loss
            'cfg_loss_importance': 1.0, # means normal weight for CFG loss
            'dfg_loss_importance': 1.0 # means normal weight for DFG loss
        }
        # =====================================================================
        # CONFIGURE WHICH TASKS TO USE FOR TRAINING
        # =====================================================================
        # Options: 'code', 'cg', 'cfg', 'dfg'
        # Examples:
        #   ['code'] - Only use code (no graphs)
        #   ['cg', 'cfg', 'dfg'] - Only use graphs (no code)
        #   ['code', 'cg', 'cfg', 'dfg'] - Use everything (default)
        #   ['cg'] - Only use Call Graph
        self.task_keys = [ 'cg', 'cfg', 'dfg' ]  # <-- MODIFY THIS TO CHANGE TASKS
        
        # Create a dictionary of criteria
        self.criteria = {}
        if 'code' in self.task_keys:
            pos_weights = self.calculate_pos_weights(['code'],self.train_loader, self.data_feat_dims["OUTPUT_DIM"], self.device)
        for task_key in self.task_keys:
            if 'cg' in task_key:
                self.criteria['cg'] = FocalLoss(gamma=2.0, reduction='none', alpha=0.6)
            elif 'cfg' in task_key:
                self.criteria['cfg'] = FocalLoss(gamma=2.0, reduction='none', alpha=0.55)
            elif 'dfg' in task_key:
                self.criteria['dfg'] = FocalLoss(gamma=2.0, reduction='none', alpha=0.55)
            else:
                self.criteria[task_key] = nn.BCEWithLogitsLoss(reduction='none', pos_weight=pos_weights[task_key])
        
        
        # Initialize history tracking for metrics using dict comprehension
        #self.val_metrics_keys = ['f1', 'auc', 'f1_fixed', 'acc_fixed']
        self.val_metrics_keys = ['f1', 'auc', 'f1_fixed', 'acc_fixed', 'true_pos%', 'pred_pos%']
        # Determine which branches to use based on task_keys
        self.use_code = 'code' in self.task_keys
        self.use_graph = any(task in self.task_keys for task in ['cg', 'cfg', 'dfg'])
        
        logger.info(f"[PURPLE]Training Configuration:")
        logger.info(f"[PURPLE]  Active Tasks: {self.task_keys}")
        logger.info(f"[PURPLE]  Use Code Branch: {self.use_code}")
        logger.info(f"[PURPLE]  Use Graph Branch: {self.use_graph}")
        
        self.history = {
            'train_loss': [],
            'val_loss': [],
        }
        
        # Add task-specific loss tracking
        for task in self.task_keys:
            # Use 'train_main_loss' for code task, 'train_{task}_loss' for others
            loss_key = 'train_main_loss' if task == 'code' else f'train_{task}_loss'
            self.history[loss_key] = []
        
        # Add validation metrics for each active task
        for task in self.task_keys:
            for metric in self.val_metrics_keys:
                self.history[f'val_{task}_{metric}'] = []
    # endregion
    
    # region Random Seed Setup
    def set_rand_seed(self, seed):
        """Set random seed for reproducibility across all libraries."""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.Generator().manual_seed(seed)
    # endregion

    # region Data Loading
    def load_data(self, rand_seed=42):
        """Load and prepare train/val/test data loaders with stratified split."""
        logger.info("Loading dataset...")
        try:
            self.dataset = CustomDataset(
                tokenizer=None,
                args=None,
                source="DAppSCAN",
                force_reload=False,
                load_type="both",
                rand_seed=rand_seed
            )
            logger.info(f"Dataset loaded with {len(self.dataset)} samples")

            # fetch data dimensions if available
            self.data_feat_dims["GRAPH_IN_DIMS"] = self.dataset.data_feat_dims.get("GRAPH_IN_DIMS", self.data_feat_dims["GRAPH_IN_DIMS"])
            self.data_feat_dims["EDGE_DIMS"] = self.dataset.data_feat_dims.get("EDGE_DIMS", self.data_feat_dims["EDGE_DIMS"])
            self.data_feat_dims["CODE_IN_DIM"] = self.dataset.data_feat_dims.get("CODE_IN_DIM", self.data_feat_dims["CODE_IN_DIM"])


        except Exception as e:
            logger.error(f"Failed to load dataset: {e}")
            traceback.print_exc()
            return
        
        # =====================================================================
        # Create Stratified Split
        # =====================================================================
        
        logger.info("Creating stratified train/val/test split...")
        
        # Extract stratification labels from dataset
        # We use binary indicator: has any vulnerability (excluding BENIGN)
        stratify_labels = []
        for i in range(len(self.dataset)):
            try:
                item = self.dataset[i]
                labels = item.get('labels', None)
                
                if labels is not None and isinstance(labels, torch.Tensor):
                    # Check if any vulnerability exists
                    # labels shape: (num_lines, 8) - 8 vulnerability types (no Benign column)
                    # If dataset still has 9 columns, take only first 8
                    vuln_labels = labels[:, :8] if labels.shape[1] >= 8 else labels
                    has_vuln = (vuln_labels.sum() > 0).item()
                    stratify_labels.append(int(has_vuln))
                else:
                    stratify_labels.append(0)  # No labels = benign
            except Exception as e:
                logger.warning(f"Could not extract label for sample {i}: {e}")
                stratify_labels.append(0)
        
        logger.info(f"Stratification distribution: Vulnerable={sum(stratify_labels)}, Benign={len(stratify_labels)-sum(stratify_labels)}")
        
        # Check if we have enough samples for stratification
        unique_labels = len(set(stratify_labels))
        if unique_labels < 2:
            logger.warning("Only one class found in dataset. Falling back to random split.")
            # Fallback to random split
            num_total = len(self.dataset)
            num_train = int(num_total * self.train_perc)
            num_val = int(num_total * self.val_perc)
            num_test = num_total - num_train - num_val
            
            train_dataset, val_dataset, test_dataset = random_split(
                self.dataset, 
                [num_train, num_val, num_test],
                generator=torch.Generator().manual_seed(rand_seed)
            )
        else:
            # Stratified split
            indices = list(range(len(self.dataset)))
            
            # First split: train vs temp (val+test)
            try:
                train_indices, temp_indices = train_test_split(
                    indices,
                    train_size=self.train_perc,
                    stratify=stratify_labels,
                    random_state=rand_seed
                )
                
                # Second split: val vs test from temp
                temp_labels = [stratify_labels[i] for i in temp_indices]
                val_size = self.val_perc / (1 - self.train_perc)
                
                val_indices, test_indices = train_test_split(
                    temp_indices,
                    train_size=val_size,
                    stratify=temp_labels,
                    random_state=rand_seed
                )
                
                # Create subsets
                train_dataset = Subset(self.dataset, train_indices)
                val_dataset = Subset(self.dataset, val_indices)
                test_dataset = Subset(self.dataset, test_indices)
                
                logger.info("✓ Stratified split successful")
                
            except ValueError as e:
                logger.warning(f"Stratified split failed ({e}). Falling back to random split.")
                # Fallback to random split
                num_total = len(self.dataset)
                num_train = int(num_total * self.train_perc)
                num_val = int(num_total * self.val_perc)
                num_test = num_total - num_train - num_val
                
                train_dataset, val_dataset, test_dataset = random_split(
                    self.dataset, 
                    [num_train, num_val, num_test],
                    generator=torch.Generator().manual_seed(rand_seed)
                )

        logger.info(f"[PURPLE]Data split complete:")
        logger.info(f"[PURPLE]  Train: {len(train_dataset)} samples")
        logger.info(f"[PURPLE]  Val:   {len(val_dataset)} samples")
        logger.info(f"[PURPLE]  Test:  {len(test_dataset)} samples")

        logger.info("Creating DataLoaders...")
        try:
            self.train_loader = GraphDataLoader(
                train_dataset, 
                batch_size=self.batch_size, 
                collate_fn=custom_collate,
                shuffle=True, 
                drop_last=False, 
                num_workers=self.dataloader_workers
            )

            self.val_loader = GraphDataLoader(
                val_dataset,
                batch_size=self.batch_size,
                shuffle=False, # No need to shuffle validation
                collate_fn=custom_collate,
                num_workers=self.dataloader_workers
            )

            self.test_loader = GraphDataLoader(
                test_dataset,
                batch_size=self.batch_size,
                shuffle=False, # No need to shuffle test
                collate_fn=custom_collate,
                num_workers=self.dataloader_workers
            )

            logger.info(f"DataLoaders created with batch_size={self.batch_size}")
            self.test_data_loader()
        except Exception as e:
            logger.error(f"Failed to create DataLoader: {e}")
            traceback.print_exc()
            return
    # endregion
    
    def test_data_loader(self):
        # region test
        # =====================================================================
        # Test DataLoader - Comprehensive Batch Analysis
        # =====================================================================
        
        logger.info("=" * 40)
        logger.info("Testing DataLoader - Detailed Batch Analysis (Train, Val, Test)")
        logger.info("=" * 40)
        
        # Analyze each dataset
        for dataset_name, loader in [
            ("TRAIN", self.train_loader),
            ("VALIDATION", self.val_loader),
            ("TEST", self.test_loader)
        ]:
            logger.info(f"\n{'#'*40}")
            logger.info(f"# {dataset_name} DATASET")
            logger.info(f"{'#'*40}\n")
            
            self._analyze_dataloader(loader, dataset_name)
        
        logger.info(f"\n{'='*40}")
        logger.info(f"✓ DataLoader Test Complete for All Datasets!")
        logger.info(f"{'='*40}\n")

    def _analyze_dataloader(self, loader, dataset_name):
        """Analyze a single dataloader (train/val/test)."""
        try:
            num_batches = 0
            total_graphs = 0
            
            # Track statistics across batches
            batch_stats = {
                'node_counts': {'ast_node': [], 'block': [], 'function': [], 'dfg_node': []},
                'edge_counts': [],
                'edge_type_counts': {},  # Track each edge type
                'code_lengths': [],
                'cg_lengths': [],
                'cfg_lengths': [],
                'dfg_lengths': [],
                'max_code_len': [],
                'max_cg_len': [],
                'max_cfg_len': [],
                'max_dfg_len': [],
                # Label statistics
                'label_stats': {
                    'code': {'total': 0, 'vulnerable': 0, 'per_class': [0]*8},
                    'cg': {'total': 0, 'vulnerable': 0, 'per_class': [0]*8},
                    'cfg': {'total': 0, 'vulnerable': 0, 'per_class': [0]*8},
                    'dfg': {'total': 0, 'vulnerable': 0, 'per_class': [0]*8}
                }
            }
            
            for batch_idx, batch in enumerate(loader):
                if batch is None:
                    logger.warning(f"[{dataset_name}] Batch {batch_idx}: None (skipped)")
                    continue
                
                num_batches += 1
                
                logger.info(f"\n{'='*40}")
                logger.info(f"[{dataset_name}] BATCH {batch_idx + 1}")
                logger.info(f"{'='*40}")
                
                # --- 1. Graph Information ---
                if 'graph' in batch and batch['graph'] is not None:
                    graph = batch['graph']
                    batch_size = graph.batch_size
                    total_graphs += batch_size
                    
                    logger.info(f"[GRAPH] Batch Size: {batch_size} graphs")
                    logger.info(f"[GRAPH] Total Nodes: {graph.num_nodes():,}")
                    logger.info(f"[GRAPH] Total Edges: {graph.num_edges():,}")
                    
                    # Node counts per type
                    logger.info(f"[NODES] Node Distribution:")
                    for ntype in graph.ntypes:
                        count = graph.num_nodes(ntype)
                        logger.info(f"        - {ntype:15s}: {count:6,} nodes")
                        batch_stats['node_counts'][ntype].append(count)
                    
                    batch_stats['edge_counts'].append(graph.num_edges())
                    
                    # Track edge counts by type
                    for src, etype, dst in graph.canonical_etypes:
                        edge_count = graph.num_edges((src, etype, dst))
                        edge_key = f"{src}--[{etype}]-->{dst}"
                        if edge_key not in batch_stats['edge_type_counts']:
                            batch_stats['edge_type_counts'][edge_key] = []
                        batch_stats['edge_type_counts'][edge_key].append(edge_count)
                    
                    # Edge types
                    logger.info(f"[EDGES] Edge Types ({len(graph.etypes)}): {graph.etypes}")
                else:
                    logger.warning(f"[GRAPH] No graph data in batch")
                
                # --- 2. Tensor Shapes ---
                logger.info(f"[TENSORS] Padded Tensor Shapes:")
                tensor_keys = ['code', 'labels', 'cg_labels', 'cfg_labels', 'dfg_labels']
                for key in tensor_keys:
                    if key in batch and batch[key] is not None:
                        shape = batch[key].shape
                        logger.info(f"          - {key:15s}: {str(shape):30s} | dtype: {batch[key].dtype}")
                        
                        # Track max sequence lengths
                        if key == 'code':
                            batch_stats['max_code_len'].append(shape[1])
                        elif key == 'cg_labels':
                            batch_stats['max_cg_len'].append(shape[1])
                        elif key == 'cfg_labels':
                            batch_stats['max_cfg_len'].append(shape[1])
                        elif key == 'dfg_labels':
                            batch_stats['max_dfg_len'].append(shape[1])
                
                # --- 3. Actual Lengths (Non-Padded) ---
                logger.info(f"[LENGTHS] Actual Sequence Lengths (per sample in batch):")
                length_keys = ['code_lengths', 'cg_lengths', 'cfg_lengths', 'dfg_lengths']
                for key in length_keys:
                    if key in batch and batch[key] is not None:
                        lengths = batch[key]
                        logger.info(f"          - {key:15s}: {lengths.tolist()} | mean: {lengths.float().mean():.1f} | max: {lengths.max().item()}")
                        
                        # Store for statistics
                        if key == 'code_lengths':
                            batch_stats['code_lengths'].extend(lengths.tolist())
                        elif key == 'cg_lengths':
                            batch_stats['cg_lengths'].extend(lengths.tolist())
                        elif key == 'cfg_lengths':
                            batch_stats['cfg_lengths'].extend(lengths.tolist())
                        elif key == 'dfg_lengths':
                            batch_stats['dfg_lengths'].extend(lengths.tolist())
                
                # --- 4. Edge Distribution per Type ---
                if 'graph' in batch and batch['graph'] is not None:
                    graph = batch['graph']
                    logger.info(f"[EDGE DISTRIBUTION] Edges per Type:")
                    
                    # Group edges by category
                    intra_graph_edges = {}
                    inter_graph_edges = {}
                    
                    for src, etype, dst in graph.canonical_etypes:
                        edge_count = graph.num_edges((src, etype, dst))
                        edge_key = f"{src} --[{etype}]--> {dst}"
                        
                        # Categorize edges
                        if etype in ['child_of', 'flows', 'calls', 'data_flow']:
                            intra_graph_edges[edge_key] = edge_count
                        else:
                            inter_graph_edges[edge_key] = edge_count
                    
                    # Log intra-graph edges (within same graph type)
                    if intra_graph_edges:
                        logger.info(f"        [Intra-Graph Edges]:")
                        for edge_key, count in intra_graph_edges.items():
                            logger.info(f"          - {edge_key:50s}: {count:6,} edges")
                    
                    # Log inter-graph edges (linking different graph types)
                    if inter_graph_edges:
                        logger.info(f"        [Inter-Graph Edges]:")
                        for edge_key, count in inter_graph_edges.items():
                            logger.info(f"          - {edge_key:50s}: {count:6,} edges")
                
                # --- 5. Label Distribution Analysis (All Tasks) ---
                logger.info(f"[LABEL DISTRIBUTION] Vulnerability Label Statistics:")
                
                # Helper function to analyze labels
                def analyze_label_distribution(labels, lengths, task_name, task_key):
                    if labels is None or labels.shape[1] == 0 or lengths.sum() == 0:
                        logger.info(f"        [{task_name}] No valid labels")
                        return
                    
                    # Create mask for valid (non-padded) elements
                    mask = torch.arange(labels.shape[1])[None, :] < lengths[:, None]
                    
                    # If dataset still has 9 columns, take only first 8 (exclude Benign)
                    if labels.shape[2] >= 9:
                        labels = labels[:, :, :8]
                    
                    valid_labels = labels[mask]  # Shape: (num_valid_nodes, 8)
                    
                    if valid_labels.numel() == 0:
                        logger.info(f"        [{task_name}] No valid labels after masking")
                        return
                    
                    # Overall statistics
                    num_total_elements = valid_labels.shape[0]
                    
                    # Check for vulnerabilities - any non-zero value in the 8 vulnerability types
                    num_positive_any = (valid_labels.sum(dim=1) > 0).sum().item()
                    
                    num_benign = num_total_elements - num_positive_any
                    pos_ratio = (num_positive_any / num_total_elements * 100) if num_total_elements > 0 else 0
                    
                    # Accumulate statistics
                    batch_stats['label_stats'][task_key]['total'] += num_total_elements
                    batch_stats['label_stats'][task_key]['vulnerable'] += num_positive_any
                    
                    logger.info(f"        [{task_name}] Total Nodes: {num_total_elements:,}")
                    logger.info(f"        [{task_name}] Vulnerable Nodes: {num_positive_any:,} ({pos_ratio:.2f}%)")
                    logger.info(f"        [{task_name}] Benign Nodes: {num_benign:,} ({100-pos_ratio:.2f}%)")
                    
                    # Per-class breakdown (8 vulnerability types)
                    class_names = [
                        'Reentrancy', 'Access Control', 'Arithmetic', 
                        'Unchecked Call', 'Denial of Service', 'Bad Randomness',
                        'Front Running', 'Time Manipulation'
                    ]
                    
                    logger.info(f"        [{task_name}] Per-Class Distribution:")
                    for class_idx in range(valid_labels.shape[1]):
                        class_count = (valid_labels[:, class_idx] > 0).sum().item()
                        class_pct = (class_count / num_total_elements * 100) if num_total_elements > 0 else 0
                        class_name = class_names[class_idx] if class_idx < len(class_names) else f'Class_{class_idx}'
                        
                        # Accumulate per-class counts
                        batch_stats['label_stats'][task_key]['per_class'][class_idx] += class_count
                        
                        if class_count > 0:  # Only show classes that have positive samples
                            logger.info(f"          - {class_name:20s}: {class_count:5,} ({class_pct:6.3f}%)")
                
                # Analyze all label types
                if 'labels' in batch and batch['labels'] is not None:
                    analyze_label_distribution(
                        batch['labels'], 
                        batch.get('code_lengths', torch.tensor([batch['labels'].shape[1]] * batch['labels'].shape[0])),
                        'CODE',
                        'code'
                    )
                
                if 'cg_labels' in batch and batch['cg_labels'] is not None:
                    analyze_label_distribution(
                        batch['cg_labels'],
                        batch.get('cg_lengths', torch.tensor([batch['cg_labels'].shape[1]] * batch['cg_labels'].shape[0])),
                        'CG (Call Graph)',
                        'cg'
                    )
                
                if 'cfg_labels' in batch and batch['cfg_labels'] is not None:
                    analyze_label_distribution(
                        batch['cfg_labels'],
                        batch.get('cfg_lengths', torch.tensor([batch['cfg_labels'].shape[1]] * batch['cfg_labels'].shape[0])),
                        'CFG (Control Flow)',
                        'cfg'
                    )
                
                if 'dfg_labels' in batch and batch['dfg_labels'] is not None:
                    analyze_label_distribution(
                        batch['dfg_labels'],
                        batch.get('dfg_lengths', torch.tensor([batch['dfg_labels'].shape[1]] * batch['dfg_labels'].shape[0])),
                        'DFG (Data Flow)',
                        'dfg'
                    )
            
            # --- Summary Statistics ---
            logger.info(f"\n{'='*80}")
            logger.info(f"{dataset_name} DATASET SUMMARY")
            logger.info(f"{'='*80}")
            logger.info(f"✓ Total Batches Processed: {num_batches}")
            logger.info(f"✓ Total Graphs: {total_graphs}")
            logger.info(f"✓ Samples per Batch: {self.batch_size}")
            
            # Node statistics
            logger.info(f"\n[NODE STATISTICS]")
            for ntype, counts in batch_stats['node_counts'].items():
                if counts:
                    logger.info(f"  {ntype:15s}: min={min(counts):6,} | max={max(counts):6,} | avg={sum(counts)//len(counts):6,}")
            
            # Edge statistics
            if batch_stats['edge_counts']:
                logger.info(f"\n[EDGE STATISTICS]")
                logger.info(f"  Total Edges      : min={min(batch_stats['edge_counts']):6,} | max={max(batch_stats['edge_counts']):6,} | avg={sum(batch_stats['edge_counts'])//len(batch_stats['edge_counts']):6,}")
            
            # Length statistics
            logger.info(f"\n[SEQUENCE LENGTH STATISTICS]")
            for key in ['code_lengths', 'cg_lengths', 'cfg_lengths', 'dfg_lengths']:
                if batch_stats[key]:
                    vals = batch_stats[key]
                    logger.info(f"  {key:15s}: min={min(vals):5} | max={max(vals):5} | avg={sum(vals)//len(vals):5}")
            
            # Padding statistics
            logger.info(f"\n[PADDING STATISTICS (Max Length per Batch)]")
            for key in ['max_code_len', 'max_cg_len', 'max_cfg_len', 'max_dfg_len']:
                if batch_stats[key]:
                    vals = batch_stats[key]
                    logger.info(f"  {key:15s}: min={min(vals):5} | max={max(vals):5} | avg={sum(vals)//len(vals):5}")
            
            # Edge type statistics
            logger.info(f"\n[EDGE TYPE STATISTICS]")
            if batch_stats['edge_type_counts']:
                # Categorize edges
                intra_edges = {}
                inter_edges = {}
                
                for edge_key, counts in batch_stats['edge_type_counts'].items():
                    # Check if it's an intra-graph edge
                    if any(x in edge_key for x in ['child_of', 'flows', 'calls', 'data_flow']):
                        intra_edges[edge_key] = counts
                    else:
                        inter_edges[edge_key] = counts
                
                logger.info(f"  [Intra-Graph Edges] (within same graph component):")
                for edge_key, counts in sorted(intra_edges.items()):
                    if counts:
                        logger.info(f"    {edge_key:50s}: min={min(counts):6,} | max={max(counts):6,} | avg={sum(counts)//len(counts):6,} | total={sum(counts):8,}")
                
                logger.info(f"  [Inter-Graph Edges] (linking different components):")
                for edge_key, counts in sorted(inter_edges.items()):
                    if counts:
                        logger.info(f"    {edge_key:50s}: min={min(counts):6,} | max={max(counts):6,} | avg={sum(counts)//len(counts):6,} | total={sum(counts):8,}")
            
            # Label distribution statistics (aggregated across all batches)
            logger.info(f"\n[AGGREGATED LABEL STATISTICS]")
            class_names = [
                'Reentrancy', 'Access Control', 'Arithmetic', 
                'Unchecked Call', 'Denial of Service', 'Bad Randomness',
                'Front Running', 'Time Manipulation'
            ]
            
            for task_key, task_name in [('code', 'CODE'), ('cg', 'CG'), ('cfg', 'CFG'), ('dfg', 'DFG')]:
                stats = batch_stats['label_stats'][task_key]
                if stats['total'] > 0:
                    vuln_pct = (stats['vulnerable'] / stats['total'] * 100)
                    benign_pct = 100 - vuln_pct
                    
                    logger.info(f"  [{task_name}] Task:")
                    logger.info(f"    Total Nodes: {stats['total']:,}")
                    logger.info(f"    Vulnerable:  {stats['vulnerable']:,} ({vuln_pct:.2f}%)")
                    logger.info(f"    Benign:      {stats['total'] - stats['vulnerable']:,} ({benign_pct:.2f}%)")
                    logger.info(f"    Per-Class Breakdown:")
                    
                    for class_idx, class_name in enumerate(class_names):
                        count = stats['per_class'][class_idx]
                        if count > 0:
                            pct = (count / stats['total'] * 100)
                            logger.info(f"      - {class_name:20s}: {count:6,} ({pct:6.3f}%)")
            
        except Exception as e:
            logger.error(f"Error analyzing {dataset_name} DataLoader: {e}")
            traceback.print_exc()

    def calculate_pos_weights(self, tasks, train_loader, output_dim, device):
        # region calculate_pos_weights
        # =====================================================================
        # Calculate Positive Weights
        # =====================================================================
        
        """Calculate pos_weight for BCEWithLogitsLoss for imbalanced datasets."""
        logger.info("Calculating positive label weights from training data...")
        
        # Initialize counts for each ACTIVE task only
        counts = {}
        for task_key in tasks:
            counts[task_key] = {
                'pos': torch.zeros(output_dim, device=device), 
                'neg': torch.zeros(output_dim, device=device)
            }
        
        # Map task keys to their corresponding label and length keys
        task_mapping = {
            'code': ('labels', 'code_lengths'),
            'cg': ('cg_labels', 'cg_lengths'),
            'cfg': ('cfg_labels', 'cfg_lengths'),
            'dfg': ('dfg_labels', 'dfg_lengths')
        }
        
        for batch_data in tqdm(train_loader, desc="Calculating Weights"):
            for task_key in tasks:
                label_key, len_key = task_mapping[task_key]
                labels = batch_data[label_key].to(device)
                lengths = batch_data[len_key].to(device)
                
                # Skip if no valid samples (e.g., all DFG nodes are empty)
                if labels.shape[1] == 0 or lengths.sum() == 0:
                    continue
                
                # Create mask
                mask = torch.arange(labels.shape[1], device=device)[None, :] < lengths[:, None]
                mask_expanded = mask.unsqueeze(-1).expand_as(labels)
                
                # Count positives and negatives (only non-padded)
                # We count per-class (dim=0)
                counts[task_key]['pos'] += (labels[mask_expanded] > 0).sum(dim=0)
                counts[task_key]['neg'] += (labels[mask_expanded] == 0).sum(dim=0)
                
        # Calculate weights
        weights = {}
        for task_key in tasks:
            pos_count = counts[task_key]['pos']
            neg_count = counts[task_key]['neg']
            
            ratio = neg_count / (pos_count + 1e-8)
            
            # if task_key == 'cg':
            #     pos_weight = torch.clamp(ratio, max=4.0)
            # elif task_key == 'cfg':
            #     pos_weight = torch.pow(ratio, 0.33)
            # else:
            #     # The others are fine with sqrt
            #     pos_weight = torch.sqrt(ratio)
            
            pos_weight = torch.clamp(ratio, min=1.0, max=100.0)
            pos_weight[pos_count == 0] = 1.0
            
            weights[task_key] = pos_weight
            logger.info(f"[PURPLE]Task '{task_key}' Weights: {weights[task_key].cpu().numpy().round(2)}")

        return weights

    def setup_model(self):
        # region setup_model
        # =====================================================================
        # Initialize Model
        # =====================================================================
        logger.info("Initializing model...")

        try:
            # region Models init
            self.model = CombinedModel(
                node_in_dims=self.data_feat_dims["GRAPH_IN_DIMS"],
                edge_in_dims=self.data_feat_dims["EDGE_DIMS"],
                code_in_dim=self.data_feat_dims["CODE_IN_DIM"],
                graph_hidden_dim = 96, 
                code_hidden_dim = 256,
                out_dim=self.data_feat_dims["OUTPUT_DIM"],
                use_graph=self.use_graph,  # Only use graph branch if needed
                use_code=self.use_code     # Only use code branch if needed
            ).to(self.device)
                    
            total_params = sum(p.numel() for p in self.model.parameters())
            logger.info(f"Model initialized with {total_params:,} parameters")
            logger.info(f"Model architecture:\n{self.model}")
        
        except Exception as e:
            logger.error(f"Failed to initialize model: {e}")
            traceback.print_exc()
            return
    
        # =====================================================================
        # Setup Hyperparams for Training
        # =====================================================================
        logger.info("Setting up training components...")
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', patience=self.patience, factor=0.5, verbose=True)
        self._log_model_summary()

    def _log_model_summary(self):
        # =====================================================================
        # Model Summary
        # =====================================================================
        
        """Generate and log model summary."""
        try:
            from torchinfo import summary
            logger.info("Generating model summary...")
            
            summary_batch = next(iter(self.val_loader))
            
            # Move the sample batch to the device
            batch_gpu = {}
            for key, tensor in summary_batch.items():
                if isinstance(tensor, torch.Tensor):
                    batch_gpu[key] = tensor.to(self.device)
                elif isinstance(tensor, dgl.DGLGraph):
                    batch_gpu[key] = tensor.to(self.device)
            
            logger.info("" + "=" * 40)
            logger.info("--- Model Summary ---")
            
            # Generate the summary by passing the sample batch as input_data
            summary(self.model, 
                    input_data=batch_gpu,
                    depth=8,
                    col_names=["input_size", "output_size", "num_params", "mult_adds"])
                    
            logger.info("=" * 40 + "\n")
            
        except Exception as e:
            logger.warning(f"Could not generate model summary: {e}")
            logger.info(f"Model Architecture (simple):\n{self.model}") # Fallback

    def _create_mask(self, lengths, max_len):
        """Creates a boolean mask from a tensor of lengths."""
        # lengths shape: (batch_size,)
        # max_len: int
        # Returns: (batch_size, max_len)
        return torch.arange(max_len, device=self.device)[None, :] < lengths[:, None]

    def _compute_masked_loss(self, preds, labels, lengths, criterion):
        # =====================================================================
        # Compute Masked Loss -> Loss number
        # =====================================================================

        """Computes the loss only on the non-padded elements."""
        # preds shape: (B, max_len, C)
        # labels shape: (B, max_len, C)
        # lengths shape: (B,)
        # criterion: The specific loss function to use (with pos_weight)

        # Handle empty sequences (e.g., no DFG nodes in batch)
        if preds.shape[1] == 0 or lengths.sum() == 0:
            return torch.tensor(0.0, device=self.device)

        # Ensure labels are float
        labels = labels.float()

        # Get max length from predictions
        max_len = preds.shape[1]

        # Create mask
        # mask shape: (B, max_len)
        mask = self._create_mask(lengths, max_len)

        # --- START FIX ---
        # Select only valid elements *before* passing to criterion.
        # This prevents -1 padding values from being sent to the loss function.
        # preds_masked shape: (num_active_nodes, C)
        # labels_masked shape: (num_active_nodes, C)
        preds_masked = preds[mask]
        labels_masked = labels[mask]

        if labels_masked.numel() == 0:
            # All elements in the batch were padding
            return torch.tensor(0.0, device=self.device)

        # Compute loss for all *active* elements
        # criterion has reduction='none', so loss_all is (num_active_nodes, C)
        loss_all = criterion(preds_masked, labels_masked)

        # Compute mean loss over all active elements and classes
        return loss_all.mean()
        # --- END FIX ---


    def train(self):

        # region train
        # =====================================================================
        # Training Loop
        # =====================================================================
        try:
            best_val_loss = float('inf')
            patience_counter = 0
            
            for epoch in range(self.num_epochs):
                logger.info(f"Epoch {epoch + 1}/{self.num_epochs}")
                self.model.train()
                train_loss, num_batches = 0.0, 0
                task_losses = {task: 0.0 for task in self.task_keys}
                task_map = {'code': ('code', 'labels', 'code_lengths'), 'cg': ('cg', 'cg_labels', 'cg_lengths'), 'cfg': ('cfg', 'cfg_labels', 'cfg_lengths'), 'dfg': ('dfg', 'dfg_labels', 'dfg_lengths')}
                
                self.optimizer.zero_grad()
                
                for i, batch in enumerate(tqdm(self.train_loader, desc="Training")):
                    if batch is None: continue
                    try:
                        batch_gpu = {k: v.to(self.device) if isinstance(v, (torch.Tensor, dgl.DGLGraph)) else v for k, v in batch.items()}
                        preds = self.model(batch_gpu)
                        
                        total_loss = torch.tensor(0.0, device=self.device)
                        for task in self.task_keys:
                            pkey, lkey, lenkey = task_map[task]
                            if pkey not in preds: continue
                            t_loss = self._compute_masked_loss(preds[pkey], batch_gpu[lkey], batch_gpu[lenkey], self.criteria[task])
                            
                            imp_key = f'{task}_loss_importance' if task != 'code' else 'fused_loss_importance'
                            total_loss += t_loss * self.tune_params.get(imp_key, 1.0)
                            task_losses[task] += t_loss.item()
                        
                        if total_loss.item() == 0.0: continue
                        
                        train_loss += total_loss.item()
                        (total_loss / self.accumulation_steps).backward()
                        
                        if (i + 1) % self.accumulation_steps == 0 or (i + 1) == len(self.train_loader):
                            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                            self.optimizer.step()
                            self.optimizer.zero_grad()
                        num_batches += 1
                    except Exception as e:
                        logger.warning(f"Batch error: {e}")
                        continue

                avg_train = train_loss / num_batches if num_batches > 0 else 0.0
                self.history['train_loss'].append(avg_train)
                for t in self.task_keys: self.history['train_main_loss' if t=='code' else f'train_{t}_loss'].append(task_losses[t]/num_batches if num_batches>0 else 0.0)
                
                logger.info(f"Train Loss: {avg_train:.4f}")
                val_metrics = self.run_evaluation(self.val_loader, desc="Validation")
                self.scheduler.step(val_metrics['combined_loss'])
                self.history['val_loss'].append(val_metrics['combined_loss'])
                for t in self.task_keys:
                    for m in self.val_metrics_keys: self.history[f'val_{t}_{m}'].append(val_metrics[t][m])

                self._log_train_metrics(val_metrics, val_metrics['combined_loss'])
                
                if val_metrics['combined_loss'] < best_val_loss:
                    best_val_loss = val_metrics['combined_loss']
                    patience_counter = 0
                    torch.save(self.model.state_dict(), "best_model.pth")
                    logger.info(f"[GREEN] Saved best model (loss: {best_val_loss:.4f})")
                else:
                    patience_counter += 1
                    if patience_counter >= self.patience:
                        logger.info("Early stopping triggered.")
                        break
            
            logger.info("" + "=" * 40)
            logger.info("Training Complete")
            logger.info("=" * 40)
            
            try:
                self.model.load_state_dict(torch.load("best_model.pth"))
                logger.info(f"[GREEN] Best model loaded (val_loss: {best_val_loss:.4f})")
            except FileNotFoundError:
                logger.warning("best_model.pth not found. Could not load best model.")
        except Exception as e:
            logger.error(f"Error during training: {e}")
            traceback.print_exc()
            sys.exit(1)

    def _log_train_metrics(self, metrics, avg_loss):
        logger.info(f"{'='*40}\n[GREEN] Validation Loss: {avg_loss:.4f}\n{'='*40}")
        for task in self.task_keys:
            m = metrics[task]
            logger.info(f"[GREEN] Task: {task.upper()} ====================")
            logger.info(f"[GREEN]   Macro-Avg F1:        {m['f1']:.4f}")
            logger.info(f"[GREEN]   Macro-Avg AUC:       {m['auc']:.4f}")
            logger.info(f"[GREEN]   Macro-Avg Precision: {m['prec']:.4f}")
            logger.info(f"[GREEN]   Macro-Avg Recall:    {m['rec']:.4f}")
            logger.info(f"[GREEN]   Binary F1 (@{self.fixed_threshold}):  {m['f1_fixed']:.4f}")
            logger.info(f"[GREEN]   Binary Acc (@{self.fixed_threshold}): {m['acc_fixed']:.4f}")
            logger.info(f"[GREEN]   True Vulnerable %:   {m['true_pos%']:.2f}%")
            logger.info(f"[GREEN]   Pred Vulnerable %:   {m['pred_pos%']:.2f}%")
        logger.info(f"{'='*40}\n")
        

    def run_evaluation(self, loader, desc="Validation"):
        try:
            # =====================================================================
            self.model.eval()
            total_loss, num_batches = 0.0, 0
            task_map = {'code': ('code', 'labels', 'code_lengths'), 'cg': ('cg', 'cg_labels', 'cg_lengths'), 'cfg': ('cfg', 'cfg_labels', 'cfg_lengths'), 'dfg': ('dfg', 'dfg_labels', 'dfg_lengths')}
            all_metrics = {t: {'labels': [], 'probs': [], 'loss': 0.0} for t in self.task_keys}

            with torch.no_grad():
                for batch in tqdm(loader, desc=desc):
                    if batch is None: continue
                    try:
                        batch_gpu = {k: v.to(self.device) if isinstance(v, (torch.Tensor, dgl.DGLGraph)) else v for k, v in batch.items()}
                        preds = self.model(batch_gpu)
                        batch_loss = 0.0
                        for task in self.task_keys:
                            pkey, lkey, lenkey = task_map[task]
                            if pkey not in preds: continue
                            
                            t_loss = self._compute_masked_loss(preds[pkey], batch_gpu[lkey], batch_gpu[lenkey], self.criteria[task])
                            all_metrics[task]['loss'] += t_loss.item()
                            imp_key = f'{task}_loss_importance' if task != 'code' else 'fused_loss_importance'
                            batch_loss += t_loss * self.tune_params.get(imp_key, 1.0)

                            mask = torch.arange(preds[pkey].shape[1])[None, :] < batch_gpu[lenkey].cpu()[:, None]
                            all_metrics[task]['labels'].append(batch_gpu[lkey].cpu()[mask])
                            all_metrics[task]['probs'].append(torch.sigmoid(preds[pkey]).cpu()[mask])
                        
                        total_loss += batch_loss.item()
                        num_batches += 1
                    except Exception as e:
                        logger.warning(f"Eval batch error: {e}")
                        continue

            final_metrics = {'combined_loss': total_loss / max(1, num_batches), 'confusion_matrices': {}}
            class_names = ['Reentrancy', 'Access Control', 'Arithmetic', 'Unchecked Call', 'DoS', 'Bad Randomness', 'Front Running', 'Time Manipulation']

            for task in self.task_keys:
                tm = {'loss': all_metrics[task]['loss'] / max(1, num_batches), 'per_class_metrics': {}}
                if not all_metrics[task]['probs']:
                    final_metrics[task] = {**tm, 'f1': 0, 'auc': 0, 'prec': 0, 'rec': 0, 'f1_fixed': 0, 'acc_fixed': 0, 'true_pos%': 0, 'pred_pos%': 0}
                    continue

                y_true = torch.cat(all_metrics[task]['labels']).numpy()
                y_prob = torch.cat(all_metrics[task]['probs']).numpy()
                
                # --- Per-Class Metrics ---
                f1s, aucs, precs, recs = [], [], [], []
                logger.info(f"{'='*5} Per-Class Metrics: {task.upper()}{'='*5}")
                
                for i in range(y_true.shape[1]):
                    cn = class_names[i] if i < len(class_names) else f'C{i}'
                    pos = y_true[:, i].sum()
                    if pos == 0:
                        logger.info(f"  [{cn:15s}] No positive samples")
                        continue
                    
                    try:
                        # Calculate AUC (threshold-independent)
                        auc = roc_auc_score(y_true[:, i], y_prob[:, i])
                        
                        # Calculate metrics at fixed threshold
                        y_pred_fixed = (y_prob[:, i] > self.fixed_threshold).astype(int)
                        prec_fixed = precision_score(y_true[:, i], y_pred_fixed, zero_division=0)
                        rec_fixed = recall_score(y_true[:, i], y_pred_fixed, zero_division=0)
                        f1_fixed = f1_score(y_true[:, i], y_pred_fixed, zero_division=0)
                        
                        # Count predictions
                        num_pred_pos = y_pred_fixed.sum()

                        logger.info(f"[{cn:15s}] n={int(pos):4} | F1={f1_fixed:.3f} P={prec_fixed:.3f} R={rec_fixed:.3f} AUC={auc:.3f} | Pred={num_pred_pos:4}")
                        
                        f1s.append(f1_fixed)
                        aucs.append(auc)
                        precs.append(prec_fixed)
                        recs.append(rec_fixed)
                        
                        tm['per_class_metrics'][cn] = {
                            'support': int(pos), 
                            'f1': f1_fixed, 
                            'auc': auc, 
                            'precision': prec_fixed, 
                            'recall': rec_fixed,
                            'num_predicted': int(num_pred_pos)
                        }
                    except Exception as e:
                        logger.warning(f"  [{cn:15s}] Error: {e}")
                        pass

                # --- Overall Metrics ---
                tm['f1'] = np.mean(f1s) if f1s else 0.0
                tm['auc'] = np.mean(aucs) if aucs else 0.0
                tm['prec'] = np.mean(precs) if precs else 0.0
                tm['rec'] = np.mean(recs) if recs else 0.0

                # Binary "Any Vulnerability" Metrics
                y_bin = (y_true.sum(axis=1) > 0).astype(int)
                prob_bin = 1 - np.prod(1 - y_prob, axis=1)
                
                # Use fixed threshold for binary predictions
                pred_bin_fixed = (prob_bin > self.fixed_threshold).astype(int)
                tm['acc_fixed'] = accuracy_score(y_bin, pred_bin_fixed)
                tm['f1_fixed'] = f1_score(y_bin, pred_bin_fixed, zero_division=0)
                tm['pred_pos%'] = (pred_bin_fixed.sum() / len(pred_bin_fixed)) * 100

                tm['true_pos%'] = (y_bin.sum() / len(y_bin)) * 100
                tm['cm'] = confusion_matrix(y_bin, pred_bin_fixed, labels=[0, 1])
                final_metrics['confusion_matrices'][task] = tm['cm']
                final_metrics[task] = tm

            return final_metrics
        except Exception as e:
            logger.error(f"Error during {desc}: {e}")
            traceback.print_exc()
            sys.exit(1) # exit on evaluation error
            return {}
        
    def visualize_results(self, test_metrics=None):
        # region visualize_results
        # =====================================================================
        # Comprehensive Results Visualization - Separate Files
        # =====================================================================
        """Create separate visualizations for training history and confusion matrices."""
        try:
            logger.info("Generating visualization plots...")
            
            # 1. Save training history
            if self.history['train_loss']:
                self._save_training_history()
            
            # 2. Save confusion matrices (binary and per-class)
            if test_metrics:
                self._save_confusion_matrices(test_metrics)
                self._save_per_class_confusion_matrices(test_metrics)
            
                logger.info(f"[GREEN] ✓ All visualization plots saved to {log_folder }/")
            
        except Exception as e:
            logger.error(f"Error creating visualizations: {e}")
            traceback.print_exc()
    
    def _save_training_history(self):
        """Save comprehensive 6-panel training history plot."""
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            # Set style for better readability
            sns.set_theme(style="whitegrid")
            
            fig, axes = plt.subplots(2, 3, figsize=(24, 12))
            fig.suptitle(f'Training History (Thresh={self.fixed_threshold})', 
                        fontsize=20, fontweight='bold', y=0.95)
            
            epochs = list(range(1, len(self.history['train_loss']) + 1))
            colors = {'code': '#1f77b4', 'cg': '#ff7f0e', 'cfg': '#2ca02c', 'dfg': '#d62728'}
            markers = {'code': 'o', 'cg': 's', 'cfg': '^', 'dfg': 'D'}
            
            # --- [1,1] Loss Curves ---
            ax = axes[0, 0]
            ax.plot(epochs, self.history['train_loss'], 'k--', linewidth=2, alpha=0.6, label='Total Train Loss')
            ax.plot(epochs, self.history['val_loss'], 'k-', linewidth=3, label='Total Val Loss')
            for task in self.task_keys:
                loss_key = 'train_main_loss' if task == 'code' else f'train_{task}_loss'
                ax.plot(epochs, self.history[loss_key], 
                        marker=markers[task], color=colors[task], alpha=0.5, 
                        linewidth=1.5, markersize=4, label=f'{task.upper()} Train Loss')
            ax.set_title('Loss Curves', fontweight='bold')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss')
            ax.legend()

            # --- [1,2] Macro AUC ---
            ax = axes[0, 1]
            for task in self.task_keys:
                ax.plot(epochs, self.history[f'val_{task}_auc'], 
                        marker=markers[task], color=colors[task], linewidth=2, 
                        label=f'{task.upper()} AUC')
            ax.set_title('Validation Macro-AUC', fontweight='bold')
            ax.set_ylim(0, 1.0)
            ax.set_xlabel('Epoch')
            ax.set_ylabel('AUC Score')
            ax.legend()

            # --- [1,3] Macro F1 ---
            ax = axes[0, 2]
            for task in self.task_keys:
                ax.plot(epochs, self.history[f'val_{task}_f1'], 
                        marker=markers[task], color=colors[task], linewidth=2, 
                        label=f'{task.upper()} F1')
            ax.set_title('Validation Macro-F1', fontweight='bold')
            ax.set_ylim(0, 1.0)
            ax.set_xlabel('Epoch')
            ax.set_ylabel('F1 Score')
            ax.legend()

            # --- [2,1] Binary F1 (Fixed Threshold) ---
            ax = axes[1, 0]
            for task in self.task_keys:
                ax.plot(epochs, self.history[f'val_{task}_f1_fixed'], 
                        marker=markers[task], color=colors[task], linewidth=2, 
                        label=f'{task.upper()} Bin F1')
            ax.set_title(f'Binary F1 (@{self.fixed_threshold})', fontweight='bold')
            ax.set_ylim(0, 1.0)
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Binary F1 Score')
            ax.legend()

            # --- [2,2] Binary Accuracy (Fixed Threshold) ---
            ax = axes[1, 1]
            for task in self.task_keys:
                ax.plot(epochs, self.history[f'val_{task}_acc_fixed'], 
                        marker=markers[task], color=colors[task], linewidth=2, 
                        label=f'{task.upper()} Bin Acc')
            ax.set_title(f'Binary Accuracy (@{self.fixed_threshold})', fontweight='bold')
            ax.set_ylim(0, 1.0)
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Accuracy')
            ax.legend()

            # --- [2,3] Vulnerability Ratio (Calibration Check) ---
            ax = axes[1, 2]
            for task in self.task_keys:
                # Plot predicted ratio trend
                ax.plot(epochs, self.history[f'val_{task}_pred_pos%'], 
                        marker=markers[task], color=colors[task], linewidth=2, 
                        label=f'{task.upper()} Pred %')
                # Plot ground truth baseline as dashed line
                true_pos = self.history[f'val_{task}_true_pos%'][-1]
                ax.axhline(y=true_pos, color=colors[task], linestyle='--', alpha=0.5,
                            label=f'{task.upper()} True % ({true_pos:.1f}%)')
            
            ax.set_title(f'Predicted vs True Vulnerable % (@{self.fixed_threshold})', 
                        fontweight='bold', color='darkred')
            ax.set_ylim(0, 100)
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Percentage (%)')
            ax.legend(ncol=2, fontsize=9)

            plt.tight_layout()
            plt.subplots_adjust(top=0.90)
            train_hist_path = os.path.join(log_folder , "training_history.png")
            plt.savefig(train_hist_path, dpi=300, bbox_inches='tight')
            logger.info(f"[GREEN] ✓ Comprehensive training history saved to {train_hist_path}")
            plt.close()
            
        except Exception as e:
            logger.warning(f"Could not save training history: {e}")
            traceback.print_exc()
    
    def _save_confusion_matrices(self, metrics):
        """Save binary confusion matrices (Benign vs Vulnerable) for each task."""
        try:
            import matplotlib.pyplot as plt
            from sklearn.metrics import ConfusionMatrixDisplay
            
            confusion_matrices = metrics.get('confusion_matrices', {})
            if not confusion_matrices:
                return
            
            num_tasks = len(self.task_keys)
            fig, axes = plt.subplots(1, num_tasks, figsize=(6 * num_tasks, 5))
            fig.suptitle('Binary Confusion Matrices (Benign vs Vulnerable)', 
                        fontsize=16, fontweight='bold')
            
            if num_tasks == 1:
                axes = [axes]
            
            task_label_map = {
                'code': 'Code Detection',
                'cg': 'Call Graph (CG)',
                'cfg': 'Control Flow (CFG)',
                'dfg': 'Data Flow (DFG)'
            }
            
            for ax, task_key in zip(axes, self.task_keys):
                cm = confusion_matrices.get(task_key)
                if cm is not None:
                    disp = ConfusionMatrixDisplay(
                        confusion_matrix=cm, 
                        display_labels=['Benign', 'Vulnerable']
                    )
                    disp.plot(ax=ax, cmap='Blues', values_format='d', colorbar=True)
                    
                    # Calculate metrics
                    tn, fp, fn, tp = cm.ravel()
                    total = tn + fp + fn + tp
                    accuracy = (tp + tn) / total if total > 0 else 0
                    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                    
                    task_label = task_label_map.get(task_key, task_key.upper())
                    ax.set_title(f'{task_label}\nAcc: {accuracy:.3f} | Prec: {precision:.3f} | Rec: {recall:.3f} | F1: {f1:.3f}', 
                                fontsize=11, fontweight='bold')
                    ax.set_xlabel('Predicted', fontsize=11, fontweight='bold')
                    ax.set_ylabel('True', fontsize=11, fontweight='bold')
                    
                    # Add percentage annotations
                    for i in range(2):
                        for j in range(2):
                            val = cm[i, j]
                            pct = (val / total * 100) if total > 0 else 0
                            text_color = 'white' if cm[i, j] > cm.max() / 2 else 'black'
                            ax.text(j, i + 0.15, f'({pct:.1f}%)', 
                                    ha='center', va='center', 
                                    color=text_color, fontsize=9, alpha=0.9)
                else:
                    ax.text(0.5, 0.5, f'No Data for {task_key}', 
                            ha='center', va='center', fontsize=12, color='red')
                    ax.set_xticks([])
                    ax.set_yticks([])
            
            plt.tight_layout()
            cm_path = os.path.join(log_folder , "binary_confusion_matrices.png")
            plt.savefig(cm_path, dpi=300, bbox_inches='tight')
            logger.info(f"[GREEN] ✓ Binary confusion matrices saved to {cm_path}")
            plt.close()
            
        except Exception as e:
            logger.warning(f"Could not save binary confusion matrices: {e}")
            traceback.print_exc()
    
    def _save_per_class_confusion_matrices(self, metrics):
        """Save fine-grained confusion matrices for each vulnerability class."""
        try:
            import matplotlib.pyplot as plt
            from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
            
            class_names = [
                'Reentrancy', 'Access Control', 'Arithmetic', 
                'Unchecked Call', 'Denial of Service', 'Bad Randomness',
                'Front Running', 'Time Manipulation'
            ]
            
            task_label_map = {
                'code': 'Code Detection',
                'cg': 'Call Graph (CG)',
                'cfg': 'Control Flow (CFG)',
                'dfg': 'Data Flow (DFG)'
            }
            
            # For each task, create a multi-panel figure showing confusion matrix for each class
            for task_key in self.task_keys:
                task_metrics = metrics.get(task_key, {})
                per_class_data = task_metrics.get('per_class_metrics', {})
                
                if not per_class_data:
                    logger.warning(f"No per-class data for task {task_key}, skipping fine-grained confusion matrices")
                    continue
                
                # Count how many classes have data
                classes_with_data = [cn for cn in class_names if cn in per_class_data and per_class_data[cn].get('support', 0) > 0]
                
                if not classes_with_data:
                    logger.warning(f"No classes with positive samples for task {task_key}")
                    continue
                
                # Create grid layout: 2 rows x 4 columns for 8 classes
                fig, axes = plt.subplots(2, 4, figsize=(20, 10))
                fig.suptitle(f'Per-Class Confusion Matrices - {task_label_map.get(task_key, task_key.upper())}', 
                            fontsize=18, fontweight='bold')
                axes = axes.flatten()
                
                for idx, class_name in enumerate(class_names):
                    ax = axes[idx]
                    
                    if class_name not in per_class_data or per_class_data[class_name].get('support', 0) == 0:
                        # No data for this class
                        ax.text(0.5, 0.5, f'{class_name}\n(No Positive Samples)', 
                                ha='center', va='center', fontsize=11, color='gray')
                        ax.set_xticks([])
                        ax.set_yticks([])
                        ax.spines['top'].set_visible(False)
                        ax.spines['right'].set_visible(False)
                        ax.spines['bottom'].set_visible(False)
                        ax.spines['left'].set_visible(False)
                        continue
                    
                    class_info = per_class_data[class_name]
                    
                    # Get confusion matrix from stored predictions
                    # We need to recompute from the stored labels and predictions
                    # For now, create a simple 2x2 matrix from precision/recall
                    support = class_info.get('support', 0)
                    precision = class_info.get('precision', 0)
                    recall = class_info.get('recall', 0)
                    
                    # Estimate confusion matrix values
                    # TP = recall * support
                    # FP = TP / precision - TP (if precision > 0)
                    # FN = support - TP
                    # We don't have exact TN, but can show relative proportions
                    
                    tp = int(recall * support) if recall > 0 else 0
                    fn = support - tp
                    fp = int(tp / precision - tp) if precision > 0 and tp > 0 else 0
                    
                    # Create simplified confusion matrix
                    cm = np.array([[0, fp], [fn, tp]])  # We don't show TN for per-class
                    
                    # Display
                    disp = ConfusionMatrixDisplay(
                        confusion_matrix=cm,
                        display_labels=['Negative', 'Positive']
                    )
                    disp.plot(ax=ax, cmap='Oranges', values_format='d', colorbar=False)
                    
                    # Add metrics to title
                    f1 = class_info.get('f1', 0)
                    auc = class_info.get('auc', 0)
                    
                    ax.set_title(f'{class_name}\n(n={support}) F1={f1:.3f} AUC={auc:.3f}', 
                                fontsize=10, fontweight='bold')
                    ax.set_xlabel('Predicted', fontsize=9)
                    ax.set_ylabel('True', fontsize=9)
                
                plt.tight_layout()
                safe_task_name = task_key.replace('/', '_')
                perclass_path = os.path.join(log_folder , f"perclass_confusion_{safe_task_name}.png")
                plt.savefig(perclass_path, dpi=300, bbox_inches='tight')
                logger.info(f"[GREEN] ✓ Per-class confusion matrices for {task_key} saved to {perclass_path}")
                plt.close()
            
        except Exception as e:
            logger.warning(f"Could not save per-class confusion matrices: {e}")
            traceback.print_exc()
    # endregion

    def test(self):
        test_metrics = self.run_evaluation(loader=self.test_loader, desc="Test")
        avg_loss = test_metrics['combined_loss']
        self._log_train_metrics(test_metrics, avg_loss)
        self.visualize_results(test_metrics)

def main():
    trainer = Trainer()
    trainer.setup_model()
    trainer.train()
    trainer.visualize_results()
    trainer.test()


if __name__ == "__main__":
    main()