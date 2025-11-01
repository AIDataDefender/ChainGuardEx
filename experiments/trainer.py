import logging
import sys
import os
import random
import traceback

import dgl
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
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
    confusion_matrix
)

try:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    sys.path.append(parent_dir)
    from experiments.dataset import CustomDataset, custom_collate
    from experiments.utils.logger import setup_logger
    from experiments.models.baseline2 import CombinedModel 
            
except ImportError:
    from dataset import CustomDataset, custom_collate
    from utils.logger import setup_logger
    from models.baseline2 import CombinedModel 
            
            
# Initialize logger for training process
logger = setup_logger("Logs/trainer.log", logging.INFO)


class Trainer:
    # region Initialization
    def __init__(self,
                batch_size=4, 
                num_epochs=10, 
                learning_rate=0.001,
                patience=5,
                hidden_dim=256, 
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
    
        logger.info("=" * 40)
        logger.info("Starting Vulnerability Detection Model Training")
        logger.info("=" * 40)
        logger.info(f"[PURPLE]Device: {self.device}")
        logger.info(f"[PURPLE]Batch size: {self.batch_size}")
        logger.info(f"[PURPLE]Num epochs: {self.num_epochs}")
        logger.info(f"[PURPLE]Learning rate: {self.learning_rate}")

        self.dataset = None
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        
        # Data feature dimensions: AST, CFG, CG, DFG graphs + CodeBERT embeddings
        self.data_feat_dims = {
            "GRAPH_IN_DIMS": { # SUBJECT TO CHANGE 
                "ast_node": 55,      # AST node feature dimension
                "block": 33,         # CFG block feature dimension
                "function": 57,      # CG function feature dimension
                "dfg_node": 41       # DFG node feature dimension (node_type + 3 hashes + src_length)
            },
            "CODE_IN_DIM": 768,      # SUBJECT TO CHANGE # CodeBERT embedding dimension
            "HIDDEN_DIM": hidden_dim,  # Model hidden layer max dimension -> will variable 
            "OUTPUT_DIM": 9          # Number of vulnerability labels
        }

        self.train_perc = 0.8 # 80% for training
        self.val_perc = 0.1 # 10% for validation
        # 10% for testing is implied (1.0 - 0.8 - 0.1)
        # Calculate the number of samples for train and val
        
        self.load_data(rand_seed=rand_seed)

        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.criteria = {}

        self.tune_params = {
            'fused_loss_importance': 1.0, # means normal weight for fused loss
            'cg_loss_importance': 1.0, # means downweight CG loss
            'cfg_loss_importance': 1.0, # means normal weight for CFG loss
            'dfg_loss_importance': 1.0 # means normal weight for DFG loss
        }
        
        # Initialize history tracking for metrics using dict comprehension
        self.val_metrics_keys = ['f1', 'auc', 'acc']
        
        # =====================================================================
        # CONFIGURE WHICH TASKS TO USE FOR TRAINING
        # =====================================================================
        # Options: 'code', 'cg', 'cfg', 'dfg'
        # Examples:
        #   ['code'] - Only use code (no graphs)
        #   ['cg', 'cfg', 'dfg'] - Only use graphs (no code)
        #   ['code', 'cg', 'cfg', 'dfg'] - Use everything (default)
        #   ['cg'] - Only use Call Graph
        self.task_keys = [ 'code', 'cg', 'cfg', 'dfg' ]  # <-- MODIFY THIS TO CHANGE TASKS
        
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
                    # Check if any vulnerability exists (excluding last BENIGN label)
                    # labels shape: (num_lines, num_vuln_types)
                    has_vuln = (labels[:, :-1].sum() > 0).item()  # Exclude BENIGN column
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
        # Test DataLoader
        # =====================================================================
        
        logger.info("Testing first batch...")
        try:
            batch = next(iter(self.train_loader))
            logger.info("✓ Batch received successfully")
            
            # --- 1. Check graph ---
            if 'graph' in batch and batch['graph'] is not None:
                graph = batch['graph']
                logger.info(f"Graph type: {type(graph)}")
                logger.info(f"Graphs in batch: {graph.batch_size}")
                logger.info(f"Total nodes: {graph.num_nodes()}")
                logger.info(f"Total edges: {graph.num_edges()}")
                logger.info(f"Node types: {graph.ntypes}")
                logger.info(f"Edge types: {graph.etypes}")
                
                # Print node counts per type
                for ntype in graph.ntypes:
                    logger.info(f"  Nodes of type '{ntype}': {graph.num_nodes(ntype)}")
            
            # --- 2. Check padded tensors ---
            tensor_keys = ['code', 'labels', 'cg_labels', 'cfg_labels', 'dfg_labels']
            for key in tensor_keys:
                if key in batch and batch[key] is not None:
                    logger.info(f"{key} shape: {batch[key].shape}")

            # --- 3. Check length tensors ---
            length_keys = ['code_lengths', 'cg_lengths', 'cfg_lengths', 'dfg_lengths']
            for key in length_keys:
                if key in batch and batch[key] is not None:
                    logger.info(f"{key} shape: {batch[key].shape}")
                    
        except Exception as e:
            logger.error(f"Error getting first batch: {e}")
            traceback.print_exc()

        logger.info("Testing iteration over multiple batches...")
        try:
            num_batches = 0
            total_graphs = 0
            
            for batch in self.train_loader:
                if batch is None:
                    print(f"   ! Warning: Batch was None")
                    continue
                
                num_batches += 1
                if 'graph' in batch and batch['graph'] is not None:
                    batch_size = batch['graph'].batch_size
                    total_graphs += batch_size

            logger.info(f"✓ Successfully iterated over {num_batches} batches")
            logger.info(f"✓ Processed {total_graphs} graphs total")

        except Exception as e:
            logger.error(f"Error during iteration: {e}")
            traceback.print_exc()

    def calculate_pos_weights(self, train_loader, output_dim, device):
        # region calculate_pos_weights
        # =====================================================================
        # Calculate Positive Weights
        # =====================================================================
        
        """Calculate pos_weight for BCEWithLogitsLoss for imbalanced datasets."""
        logger.info("Calculating positive label weights from training data...")
        
        # Initialize counts for each ACTIVE task only
        counts = {}
        for task_key in self.task_keys:
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
            for task_key in self.task_keys:
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
        for task_key in self.task_keys:
            pos_count = counts[task_key]['pos']
            neg_count = counts[task_key]['neg']
            
            ratio = neg_count / (pos_count + 1e-8)
            
            if task_key == 'cg':
                pos_weight = torch.clamp(ratio, max=4.0)
            elif task_key == 'cfg':
                pos_weight = torch.pow(ratio, 0.33)
            else:
                # The others are fine with sqrt
                pos_weight = torch.sqrt(ratio)
            
            pos_weight = torch.clamp(pos_weight, max=100.0)
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
                graph_in_dims=self.data_feat_dims["GRAPH_IN_DIMS"],
                code_in_dim=self.data_feat_dims["CODE_IN_DIM"],
                hidden_dim=self.data_feat_dims["HIDDEN_DIM"],
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
        # Create a dictionary of criteria
        pos_weights = self.calculate_pos_weights(self.train_loader, self.data_feat_dims["OUTPUT_DIM"], self.device)

        # Only create criteria for active tasks
        self.criteria = {}
        for task_key in self.task_keys:
            self.criteria[task_key] = nn.BCEWithLogitsLoss(reduction='none', pos_weight=pos_weights[task_key])
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 
            'min',  # Track validation loss
            patience=self.patience, 
            factor=0.5,
            verbose=True
        )
        
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
        
        # Reshape mask to match preds/labels
        # mask_expanded shape: (B, max_len, C)
        mask_expanded = mask.unsqueeze(-1).expand_as(preds)
        
        # Compute loss for all elements
        # loss shape: (B, max_len, C)
        loss_all = criterion(preds, labels)
        
        # Apply mask
        masked_loss = loss_all * mask_expanded
        
        # Compute mean loss *only* over non-padded elements
        # We sum all losses and divide by the total number of non-padded elements
        total_loss = masked_loss.sum()
        num_active_elements = mask_expanded.sum()
        
        if num_active_elements == 0:
            return torch.tensor(0.0, device=self.device)
            
        return total_loss / num_active_elements


    def train(self):

        # region train
        # =====================================================================
        # Training Loop
        # =====================================================================
        try:
            best_val_loss = float('inf')
            patience = self.patience
            patience_counter = 0

            logger.info("" + "=" * 40)
            logger.info("Starting Training")
            logger.info("=" * 40)
            
            for epoch in range(self.num_epochs):
                logger.info(f"Epoch {epoch + 1}/{self.num_epochs}")
                logger.info("-" * 40)
                
                # Training
                self.model.train()
                train_loss = 0.0
                # Dynamic task loss tracking
                task_losses = {task: 0.0 for task in self.task_keys}
                num_train_batches = 0
                
                # Task mapping
                task_mapping = {
                    'code': ('code', 'labels', 'code_lengths'),
                    'cg': ('cg', 'cg_labels', 'cg_lengths'),
                    'cfg': ('cfg', 'cfg_labels', 'cfg_lengths'),
                    'dfg': ('dfg', 'dfg_labels', 'dfg_lengths')
                }
                
                for batch_data in tqdm(self.train_loader, desc="Training"):
                    # Skip empty batches
                    if batch_data is None:
                        logger.warning("Skipping empty batch")
                        continue
                    
                    try:
                        # Move all data to GPU in a new dict
                        batch_gpu = {}
                        for key, tensor in batch_data.items():
                            if isinstance(tensor, torch.Tensor):
                                batch_gpu[key] = tensor.to(self.device)
                            elif isinstance(tensor, dgl.DGLGraph):
                                batch_gpu[key] = tensor.to(self.device)
                            else:
                                batch_gpu[key] = tensor
                        
                        self.optimizer.zero_grad()
                        
                        preds = self.model(batch_gpu)
                        
                        # Compute losses only for active tasks
                        total_loss = torch.tensor(0.0, device=self.device)
                        
                        for task_key in self.task_keys:
                            pred_key, label_key, len_key = task_mapping[task_key]
                            
                            # Skip if prediction not available (shouldn't happen with proper config)
                            if pred_key not in preds:
                                continue
                            
                            task_loss = self._compute_masked_loss(
                                preds[pred_key],
                                batch_gpu[label_key],
                                batch_gpu[len_key],
                                self.criteria[task_key]
                            )
                            
                            # Get importance weight for this task
                            importance_key = f'{task_key}_loss_importance' if task_key != 'code' else 'fused_loss_importance'
                            importance = self.tune_params.get(importance_key, 1.0)
                            
                            total_loss += importance * task_loss
                            task_losses[task_key] += task_loss.item()
                        
                        total_loss.backward()
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                        self.optimizer.step()
                        
                        train_loss += total_loss.item()
                        num_train_batches += 1
                    
                    except Exception as e:
                        logger.warning(f"Error in training batch: {e}")
                        traceback.print_exc()
                        continue
                
                avg_train_loss = train_loss / num_train_batches if num_train_batches > 0 else 0.0
                
                # Compute average task losses
                avg_task_losses = {task: task_losses[task] / num_train_batches if num_train_batches > 0 else 0.0 
                                   for task in self.task_keys}
                
                # Build dynamic log message
                task_loss_strs = [f"{task.upper()}: {avg_task_losses[task]:.4f}" for task in self.task_keys]
                logger.info(f"Training Loss: {avg_train_loss:.4f} | " + " | ".join(task_loss_strs))
                
                # Track training metrics in history
                self.history['train_loss'].append(avg_train_loss)
                for task in self.task_keys:
                    loss_key = 'train_main_loss' if task == 'code' else f'train_{task}_loss'
                    self.history[loss_key].append(avg_task_losses[task])
                
                val_metrics = self.run_evaluation(self.val_loader, desc="Validation")
                avg_val_loss = val_metrics['combined_loss']
                
                # Track validation metrics in history using loop
                self.history['val_loss'].append(avg_val_loss)
                for task in self.task_keys:
                    for metric in self.val_metrics_keys:
                        self.history[f'val_{task}_{metric}'].append(val_metrics[task][metric])
                
                # Log validation metrics
                self._log_train_metrics(val_metrics, avg_val_loss)
                # Learning rate scheduling
                self.scheduler.step(avg_val_loss) # Use validation loss
                
                # Early stopping based on validation loss
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    patience_counter = 0
                    # Save best model
                    torch.save(self.model.state_dict(), "best_model.pth")
                    logger.info(f"[GREEN] Best model saved (val_loss: {best_val_loss:.4f})")
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        logger.info(f"[YELLOW] Early stopping at epoch {epoch + 1}")
                        break
            
            logger.info("" + "=" * 40)
            logger.info("Training Complete")
            logger.info("=" * 40)
            
            # Load best model
            try:
                self.model.load_state_dict(torch.load("best_model.pth"))
                logger.info(f"[GREEN] Best model loaded (val_loss: {best_val_loss:.4f})")
            except FileNotFoundError:
                logger.warning("best_model.pth not found. Could not load best model.")
        except Exception as e:
            logger.error(f"Error during training: {e}")
            traceback.print_exc()
            sys.exit(1) # exit on training error


    def _log_train_metrics(self, metrics, avg_loss):
        logger.info(f"[GREEN] Validation Loss: {avg_loss:.4f}")
        for task_key in self.task_keys:
            logger.info(f"[GREEN]  Val Task {task_key} -> F1: {metrics[task_key]['f1']:.4f}, AUC: {metrics[task_key]['auc']:.4f}")
            logger.info(f"[GREEN]  True Pos %: {metrics[task_key]['true_pos%']:.4f}% | Pred Pos %: {metrics[task_key]['pred_pos%']:.4f}% (at Thresh={metrics[task_key]['best_threshold']:.4f})")
        

    def run_evaluation(self, loader, desc="Validation"):
        """Runs an evaluation loop (validation or test)."""
        try:
            logger.info(f"Running {desc}...")
            self.model.eval()
            
            total_loss = 0.0
            num_batches = 0
            
            # Task mapping
            task_mapping = {
                'code': ('code', 'labels', 'code_lengths'),
                'cg': ('cg', 'cg_labels', 'cg_lengths'),
                'cfg': ('cfg', 'cfg_labels', 'cfg_lengths'),
                'dfg': ('dfg', 'dfg_labels', 'dfg_lengths')
            }
            
            # Store all unpadded predictions and labels for sklearn - only for active tasks
            all_metrics = {task: {'labels': [], 'probs': [], 'loss': 0.0} for task in self.task_keys}
            
            with torch.no_grad():
                for batch_data in tqdm(loader, desc=desc):
                    if batch_data is None:
                        continue
                    
                    try:
                        # Move all data to GPU
                        batch_gpu = {}
                        for key, tensor in batch_data.items():
                            if isinstance(tensor, torch.Tensor):
                                batch_gpu[key] = tensor.to(self.device)
                            elif isinstance(tensor, dgl.DGLGraph):
                                batch_gpu[key] = tensor.to(self.device)
                            else:
                                batch_gpu[key] = tensor
                        
                        # Forward pass
                        preds = self.model(batch_gpu)
                        
                        batch_total_loss = 0.0

                        # Process only active tasks
                        for task_key in self.task_keys:
                            pred_key, label_key, len_key = task_mapping[task_key]
                            
                            # Skip if prediction not available
                            if pred_key not in preds:
                                continue
                            
                            preds_logits = preds[pred_key]
                            labels_true = batch_gpu[label_key]
                            lengths = batch_gpu[len_key]
                            
                            # Skip if no valid samples (e.g., all DFG nodes are empty)
                            if preds_logits.shape[1] == 0 or lengths.sum() == 0:
                                continue

                            # --- Masked Loss Calculation ---
                            batch_task_loss = self._compute_masked_loss(
                                preds_logits, 
                                labels_true, 
                                lengths,
                                self.criteria[task_key] # Pass correct criterion
                            )
                            
                            all_metrics[task_key]['loss'] += batch_task_loss.item()
                            
                            # Get importance weight for this task
                            importance_key = f'{task_key}_loss_importance' if task_key != 'code' else 'fused_loss_importance'
                            importance = self.tune_params.get(importance_key, 1.0)
                            batch_total_loss += importance * batch_task_loss

                            # --- Prepare for Sklearn Metrics ---
                            preds_probs = torch.sigmoid(preds_logits)

                            # Move to CPU for sklearn
                            labels_true_cpu = labels_true.cpu()
                            preds_probs_cpu = preds_probs.cpu()
                            
                            # Create mask on CPU
                            mask_cpu = torch.arange(preds_logits.shape[1])[None, :] < lengths.cpu()[:, None]

                            all_metrics[task_key]['labels'].append(labels_true_cpu[mask_cpu])
                            all_metrics[task_key]['probs'].append(preds_probs_cpu[mask_cpu])

                        total_loss += batch_total_loss.item()
                        num_batches += 1
                    
                    except Exception as e:
                        logger.warning(f"Error in {desc} batch: {e}")
                        traceback.print_exc()
                        continue

            avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
            
            # --- Calculate Final Metrics ---
            final_metrics = {'combined_loss': avg_loss, 'confusion_matrices': {}}
            
            for task_key in self.task_keys:
                task_metrics = {}
                try:
                    if not all_metrics[task_key]['probs']:
                        logger.warning(f"No valid data to compute metrics for task {task_key}.")
                        task_metrics.update({'acc': 0, 'prec': 0, 'rec': 0, 'f1': 0, 'auc': 0, 'loss': 0, 'true_pos%': 0, 'pred_pos%': 0, 'best_threshold': 0.5, 'cm': None})
                        final_metrics[task_key] = task_metrics
                        continue

                    all_labels = torch.cat(all_metrics[task_key]['labels']).numpy().flatten()
                    all_probs = torch.cat(all_metrics[task_key]['probs']).numpy().flatten()
                    task_metrics['loss'] = all_metrics[task_key]['loss'] / num_batches
                    
                    if len(all_labels) == 0:
                        logger.warning(f"No data for task {task_key}, skipping metrics.")
                        task_metrics.update({'acc': 0, 'prec': 0, 'rec': 0, 'f1': 0, 'auc': 0, 'true_pos%': 0, 'pred_pos%': 0, 'best_threshold': 0.5, 'cm': None})
                    else:
                        precision, recall, thresholds = precision_recall_curve(all_labels, all_probs)
                        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
                        # Find the threshold that gives the max F1
                        best_f1_idx = np.argmax(f1_scores)
                        best_threshold = thresholds[best_f1_idx]
                        
                        # Get binary predictions using the *best* threshold
                        all_preds_bin = (all_probs > best_threshold).astype(int)

                        task_metrics['best_threshold'] = best_threshold
                        task_metrics['acc'] = accuracy_score(all_labels, all_preds_bin)
                        task_metrics['prec'] = precision_score(all_labels, all_preds_bin, zero_division=0)
                        task_metrics['rec'] = recall_score(all_labels, all_preds_bin, zero_division=0)
                        task_metrics['f1'] = f1_scores[best_f1_idx]
                        
                        try:
                            # AUC uses probabilities, so it's not affected by threshold
                            task_metrics['auc'] = roc_auc_score(all_labels, all_probs)
                        except ValueError:
                            task_metrics['auc'] = float('nan')
                        
                        task_metrics['true_pos%'] = (all_labels.sum() / all_labels.size) * 100
                        task_metrics['pred_pos%'] = (all_preds_bin.sum() / all_preds_bin.size) * 100
                        
                        # Compute confusion matrix
                        cm = confusion_matrix(all_labels, all_preds_bin, labels=[0, 1])
                        task_metrics['cm'] = cm
                        final_metrics['confusion_matrices'][task_key] = cm

                    final_metrics[task_key] = task_metrics
                except Exception as e:
                    logger.error(f"Could not compute metrics for task {task_key}: {e}")
                    final_metrics[task_key] = {'loss': 0, 'acc': 0, 'prec': 0, 'rec': 0, 'f1': 0, 'auc': 0, 'true_pos%': 0, 'pred_pos%': 0, 'best_threshold': 0.5, 'cm': None}

            return final_metrics
        except Exception as e:
            logger.error(f"Error during {desc}: {e}")
            traceback.print_exc()
            sys.exit(1) # exit on evaluation error
            return {}
        
    def visualize_training_history(self):
        # region visualize_training_history
        # =====================================================================
        # Visualize Training History
        # =====================================================================
        """Plot training and validation metrics over epochs."""
        try:
            import matplotlib.pyplot as plt
            
            if not self.history['train_loss']:
                logger.warning("No training history to visualize")
                return
            
            epochs = list(range(1, len(self.history['train_loss']) + 1))
            
            # Create figure with 3 subplots: Losses, F1 Scores, AUC Scores
            fig, axes = plt.subplots(1, 3, figsize=(18, 5))
            fig.suptitle('Training History - Vulnerability Detection Model', fontsize=16, fontweight='bold')
            
            # --- Subplot 1: Losses ---
            ax_loss = axes[0]
            ax_loss.plot(epochs, self.history['train_loss'], 'o-', label='Train Total Loss', linewidth=2, markersize=5)
            ax_loss.plot(epochs, self.history['val_loss'], 's-', label='Val Total Loss', linewidth=2, markersize=5)
            
            # Plot individual task losses (only for active tasks)
            task_markers = {'code': '^', 'cg': 'v', 'cfg': 'x', 'dfg': 'd'}
            for task in self.task_keys:
                loss_key = 'train_main_loss' if task == 'code' else f'train_{task}_loss'
                task_label = 'Code' if task == 'code' else task.upper()
                ax_loss.plot(epochs, self.history[loss_key], task_markers.get(task, 'o') + '--', 
                           label=f'Train {task_label} Loss', linewidth=1.5, alpha=0.7)
            
            ax_loss.set_xlabel('Epoch', fontsize=11, fontweight='bold')
            ax_loss.set_ylabel('Loss', fontsize=11, fontweight='bold')
            ax_loss.set_title('Loss Over Epochs', fontsize=12, fontweight='bold')
            ax_loss.legend(loc='best', fontsize=9)
            ax_loss.grid(True, alpha=0.3)
            
            # --- Subplot 2: F1 Scores ---
            ax_f1 = axes[1]
            for task in self.task_keys:
                task_label = 'Code' if task == 'code' else task.upper()
                ax_f1.plot(epochs, self.history[f'val_{task}_f1'], task_markers.get(task, 'o') + '-', 
                          label=f'{task_label} Task F1', linewidth=2, markersize=5)
            
            ax_f1.set_xlabel('Epoch', fontsize=11, fontweight='bold')
            ax_f1.set_ylabel('F1 Score', fontsize=11, fontweight='bold')
            ax_f1.set_title('F1 Scores Over Epochs', fontsize=12, fontweight='bold')
            ax_f1.set_ylim([0, 1.05])
            ax_f1.legend(loc='best', fontsize=10)
            ax_f1.grid(True, alpha=0.3)
            
            # --- Subplot 3: AUC Scores ---
            ax_auc = axes[2]
            for task in self.task_keys:
                task_label = 'Code' if task == 'code' else task.upper()
                ax_auc.plot(epochs, self.history[f'val_{task}_auc'], task_markers.get(task, 'o') + '-', 
                           label=f'{task_label} Task AUC', linewidth=2, markersize=5)
            
            ax_auc.set_xlabel('Epoch', fontsize=11, fontweight='bold')
            ax_auc.set_ylabel('AUC Score', fontsize=11, fontweight='bold')
            ax_auc.set_title('AUC Scores Over Epochs', fontsize=12, fontweight='bold')
            ax_auc.set_ylim([0, 1.05])
            ax_auc.legend(loc='best', fontsize=10)
            ax_auc.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Save plot
            plot_path = "Logs/training_history.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            logger.info(f"[GREEN] Training history plot saved to {plot_path}")
            
            plt.show()
            
        except ImportError:
            logger.warning("matplotlib not installed. Cannot visualize training history.")
        except Exception as e:
            logger.error(f"Error visualizing training history: {e}")
            traceback.print_exc()
    # endregion

    def visualize_confusion_matrices(self, metrics):
        # region visualize_confusion_matrices
        # =====================================================================
        # Visualize Confusion Matrices
        # =====================================================================
        """Plot confusion matrices for all tasks (code, cg, cfg)."""
        try:
            import matplotlib.pyplot as plt
            from sklearn.metrics import ConfusionMatrixDisplay
            
            confusion_matrices = metrics.get('confusion_matrices', {})
            
            if not confusion_matrices:
                logger.warning("No confusion matrices to visualize")
                return
            
            # Create dynamic subplots based on number of tasks
            num_tasks = len(self.task_keys)
            fig, axes = plt.subplots(1, num_tasks, figsize=(5 * num_tasks, 4))
            fig.suptitle('Confusion Matrices - Test Set', fontsize=16, fontweight='bold')
            
            # Handle single task case (axes is not array)
            if num_tasks == 1:
                axes = [axes]
            
            task_label_map = {
                'code': 'Code Detection',
                'cg': 'Call Graph (CG)',
                'cfg': 'Control Flow (CFG)',
                'dfg': 'Data Flow (DFG)'
            }
            
            for idx, (task_key, ax) in enumerate(zip(self.task_keys, axes)):
                cm = confusion_matrices.get(task_key)
                
                if cm is None:
                    logger.warning(f"No confusion matrix for task {task_key}")
                    ax.text(0.5, 0.5, f'No Data\nfor {task_key}', 
                            ha='center', va='center', fontsize=12, color='red')
                    ax.set_xticks([])
                    ax.set_yticks([])
                    continue
                
                # Create display
                disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Normal', 'Vulnerable'])
                disp.plot(ax=ax, cmap='Blues', values_format='d')
                
                task_label = task_label_map.get(task_key, task_key.upper())
                ax.set_title(f'{task_label} Task', fontsize=12, fontweight='bold')
                ax.set_xlabel('Predicted Label', fontsize=10, fontweight='bold')
                ax.set_ylabel('True Label', fontsize=10, fontweight='bold')
            
            plt.tight_layout()
            
            # Save plot
            plot_path = "Logs/confusion_matrices.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            logger.info(f"✓ Confusion matrices plot saved to {plot_path}")

        except ImportError:
            logger.warning("matplotlib or sklearn ConfusionMatrixDisplay not available.")
        except Exception as e:
            logger.error(f"Error visualizing confusion matrices: {e}")
            traceback.print_exc()
    # endregion

    def test(self):
        test_metrics = self.run_evaluation(loader=self.test_loader, desc="Test")
        avg_loss = test_metrics['combined_loss']
        self._log_train_metrics(test_metrics, avg_loss)
        self.visualize_confusion_matrices(test_metrics)

def main():
    trainer = Trainer()
    trainer.setup_model()
    trainer.train()
    trainer.visualize_training_history()
    trainer.test()


if __name__ == "__main__":
    main()