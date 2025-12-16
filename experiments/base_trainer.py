from datetime import datetime
import logging
from pathlib import Path
import sys
import os
import random

import dgl
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import copy

from tqdm import tqdm
from dgl.dataloading import GraphDataLoader
from torch.utils.data import Subset
from sklearn.metrics import (
    roc_auc_score,
    f1_score,
    precision_score,
    classification_report,
)

try:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    sys.path.append(parent_dir)
    from experiments.dataset import CustomDataset, custom_collate
    from experiments.the_utils.logger import setup_logger
except ImportError:
    from dataset import CustomDataset, custom_collate
    from the_utils.logger import setup_logger

os.environ["DGLBACKEND"] = "pytorch"
os.environ["TOKENIZERS_PARALLELISM"] = "false"


class PCGrad:
    """
    Gradient Surgery for Multi-Task Learning: Projects conflicting gradients
    onto each other's normal plane.
    """

    def __init__(self, optimizer):
        self._optim = optimizer
        self.optimizer = optimizer  # compatibility

    @property
    def param_groups(self):
        return self._optim.param_groups

    def zero_grad(self):
        return self._optim.zero_grad()

    def step(self):
        return self._optim.step()

    def _project_conflicting(self, grads, has_grads, shapes=None):
        pc_grad, num_task = copy.deepcopy(grads), len(grads)
        for g_i in pc_grad:
            random.shuffle(grads)
            for g_j in grads:
                g_i_g_j = torch.dot(g_i, g_j)
                if g_i_g_j < 0:
                    g_i -= (g_i_g_j) * g_j / (g_j.norm() ** 2)
        merged_grad = torch.zeros_like(grads[0])
        for g_i in pc_grad:
            merged_grad += g_i
        return merged_grad

    def _set_grad(self, grads):
        """
        Set gradients from the list of unflattened gradient tensors.
        grads: list of gradient tensors with proper shapes for ALL parameters
        """
        idx = 0
        for group in self._optim.param_groups:
            for p in group["params"]:
                # Assign gradient (could be zero or actual gradient)
                # Ensure gradient shape matches parameter shape
                if grads[idx].shape != p.shape:
                    raise ValueError(
                        f"Gradient shape mismatch for param {idx}: expected {p.shape}, got {grads[idx].shape}"
                    )
                p.grad = grads[idx]
                idx += 1

    def _pack_grad(self, objectives):
        grads, shapes, has_grads = [], [], []
        for obj in objectives:
            self._optim.zero_grad(set_to_none=True)
            obj.backward(retain_graph=True)
            grad, shape, has_grad = [], [], []
            for group in self._optim.param_groups:
                for p in group["params"]:
                    if p.grad is None:
                        shape.append(p.shape)
                        grad.append(torch.zeros_like(p).view(-1))
                        has_grad.append(
                            torch.zeros(p.numel(), device=p.device,
                                        dtype=torch.bool)
                        )
                        continue
                    shape.append(p.grad.shape)
                    grad.append(p.grad.clone().view(-1))
                    has_grad.append(
                        torch.ones(p.numel(), device=p.device,
                                   dtype=torch.bool)
                    )
            grads.append(torch.cat(grad))
            shapes.append(shape)
            has_grads.append(torch.cat(has_grad))
        return grads, shapes, has_grads

    def _unflatten_grad(self, grads, shapes):
        unflatten_grad, idx = [], 0
        for shape in shapes:
            length = np.prod(shape)
            unflatten_grad.append(grads[idx: idx + length].view(shape))
            idx += length
        return unflatten_grad

    def pc_backward(self, objectives):
        """
        Calculate gradients for each objective and project conflicting ones.
        """
        grads, shapes, has_grads = self._pack_grad(objectives)
        pc_grad = self._project_conflicting(grads, has_grads)
        pc_grad = self._unflatten_grad(pc_grad, shapes[0])
        self._set_grad(pc_grad)


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

    def __init__(self, alpha=0.25, gamma=2.0, pos_weight=None, reduction="none"):
        """
        Args:
            alpha (float): Weight for positive class (0-1).
                            Typical: 0.25 for positive, 0.75 for negative.
                            Use higher alpha (0.5-0.75) for rare positives.
            gamma (float): Focusing parameter (0-5). Higher = more focus on hard examples.
                            Typical: 2.0. Use 3-5 for extreme imbalance.
            pos_weight (torch.Tensor): Per-class weights for positive examples [C].
            reduction (str): 'none', 'mean', or 'sum'.
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.pos_weight = pos_weight
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

        # Clamp probabilities to avoid log(0) - use larger epsilon for stability
        eps = 1e-7
        p_clamped = torch.clamp(p, min=eps, max=1.0 - eps)

        # Positive case (y=1)
        pos_loss = (
            -self.alpha * torch.pow(1 - p_clamped,
                                    self.gamma) * torch.log(p_clamped)
        )

        # Negative case (y=0)
        neg_loss = (
            -(1 - self.alpha)
            * torch.pow(p_clamped, self.gamma)
            * torch.log(1 - p_clamped)
        )

        # Combine based on target
        focal_loss = targets * pos_loss + (1 - targets) * neg_loss

        # Apply per-class weights if provided
        if self.pos_weight is not None:
            # pos_weight shape: [C], focal_loss shape: [..., C]
            # Apply weight only to positive examples
            weight_mask = targets * \
                (self.pos_weight.to(targets.device) - 1) + 1
            focal_loss = focal_loss * weight_mask

        # Apply reduction if specified
        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss


class BaseTrainer:
    def __init__(
        self,
        stage=3,  # NEW: Pass stage to trainer
        batch_size=4,
        num_epochs=30,
        learning_rate=None,  # Auto-set based on stage
        patience=10,
        hidden_dim=256,
        rand_seed=42,
        log_folder=None,
    ):
        self.stage = int(stage)
        self.set_rand_seed(rand_seed)

        # Logging
        self.log_folder = log_folder or Path(
            f'Logs/Stage{self.stage}_{datetime.now().strftime("%Y%m%d_%H%M")}')
        os.makedirs(self.log_folder, exist_ok=True)
        self.logger = setup_logger(
            f"{self.log_folder}/trainer.log", logging.INFO)

        # GPU Detection and Diagnostics
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0)
            print(f"✓ CUDA Available: {gpu_count} GPU(s) detected")
            print(f"✓ GPU 0: {gpu_name}")
            self.device = torch.device("cuda")
        else:
            print("✗ CUDA NOT Available - Running on CPU")
            print(f"  PyTorch version: {torch.__version__}")
            print(f"  CUDA built version: {torch.version.cuda}")
            self.device = torch.device("cpu")
        
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        # Stage-specific learning rates: Stage 1/2 need higher LR due to extreme imbalance
        if learning_rate is None:
            self.learning_rate = 1e-4 if self.stage in [1, 2] else 3e-5
        else:
            self.learning_rate = learning_rate
        self.patience = patience
        self.hidden_dim = hidden_dim

        # History tracking
        self.history = {"train_loss": [],
                        "val_loss": [], "val_f1": [], "val_auc": []}

        # Mode Switching
        self.is_graph_level = (self.stage in [1, 2])
        self.logger.info(
            f"Initialized Trainer for Stage {self.stage} ({'Graph' if self.is_graph_level else 'Node'} Classification)")
        self.logger.info(f"Using device: {self.device}")

        self.embedding_dims = None
        # Initialize Dataset
        self.load_data(rand_seed)

        # Initialize Model
        self.setup_model()

    def set_rand_seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def load_data(self, rand_seed):
        self.logger.info("Loading dataset...")
        # Pass stage to CustomDataset
        self.dataset = CustomDataset(
            source="DAppSCAN", force_reload=False, rand_seed=rand_seed, stage=self.stage)
        self.embedding_dims = self.dataset.embedding_dims

        # Stratified Split (80/10/10) to keep label ratios consistent across splits.
        # For Stage 1/2: binary graph labels.
        # For Stage 3: binary proxy label = any vulnerable node in (cfg|ast).
        n_total = len(self.dataset)
        if n_total <= 0:
            raise ValueError("Empty dataset")

        def _to_binary(v):
            try:
                if isinstance(v, torch.Tensor):
                    if v.numel() != 1:
                        return 0
                    v = v.item()
                if isinstance(v, bool):
                    v = int(v)
                if isinstance(v, (int, float)):
                    return 1 if float(v) > 0.5 else 0
            except Exception:
                return 0
            return 0

        labels = []
        if self.stage in [1, 2]:
            for _, lbl in getattr(self.dataset, "dataset_label", []):
                labels.append(_to_binary(lbl))
        else:
            for _, lbl in getattr(self.dataset, "dataset_label", []):
                is_pos = 0
                if isinstance(lbl, dict):
                    cfg = lbl.get("cfg_node")
                    ast = lbl.get("ast_node")
                    try:
                        if isinstance(cfg, torch.Tensor) and cfg.numel() > 0:
                            is_pos = 1 if (cfg.sum() > 0).item() else 0
                        if not is_pos and isinstance(ast, torch.Tensor) and ast.numel() > 0:
                            is_pos = 1 if (ast.sum() > 0).item() else 0
                    except Exception:
                        is_pos = 0
                labels.append(int(is_pos))

        if len(labels) != n_total:
            self.logger.warning(
                f"Label extraction mismatch: got {len(labels)} labels for dataset size {n_total}. Falling back to random split."
            )
            labels = None

        def _stratified_indices(y, seed, train_ratio=0.8, val_ratio=0.1):
            idx0 = [i for i, v in enumerate(y) if int(v) == 0]
            idx1 = [i for i, v in enumerate(y) if int(v) == 1]
            rng = random.Random(seed)
            rng.shuffle(idx0)
            rng.shuffle(idx1)

            # Log overall balance (useful for Stage 1/2 where positives may be extremely rare).
            try:
                self.logger.info(
                    f"[SPLIT DEBUG] Total={len(y)} | pos={len(idx1)} ({(len(idx1)/len(y)*100):.3f}%) | neg={len(idx0)}"
                )
            except Exception:
                pass

            # If a class is missing, stratification is impossible.
            if len(idx0) == 0 or len(idx1) == 0:
                all_idx = list(range(len(y)))
                rng.shuffle(all_idx)
                tr = int(len(all_idx) * train_ratio)
                va = int(len(all_idx) * val_ratio)
                return all_idx[:tr], all_idx[tr: tr + va], all_idx[tr + va:]

            def split_class(idxs):
                """Split a single class list into train/val/test.

                Key property: if the class has >=3 samples, guarantee at least
                1 in val and 1 in test so val metrics (e.g., F1) are meaningful.
                """
                n = len(idxs)
                if n <= 0:
                    return [], [], []
                if n == 1:
                    return idxs[:1], [], []
                if n == 2:
                    # Prefer train+val so validation sees the rare class.
                    return idxs[:1], idxs[1:2], []

                # n >= 3
                tr = int(n * train_ratio)
                va = int(n * val_ratio)
                te = n - tr - va

                # Ensure at least 1 in each split where possible.
                va = max(1, va)
                te = max(1, te)
                tr = n - va - te
                if tr <= 0:
                    tr = 1
                    # take from the larger of (va, te)
                    if va >= te and va > 1:
                        va -= 1
                    elif te > 1:
                        te -= 1
                    else:
                        # Worst-case: push everything into train+val
                        te = 0
                        va = n - tr

                return idxs[:tr], idxs[tr: tr + va], idxs[tr + va: tr + va + te]

            tr0, va0, te0 = split_class(idx0)
            tr1, va1, te1 = split_class(idx1)

            train_idx = tr0 + tr1
            val_idx = va0 + va1
            test_idx = te0 + te1
            rng.shuffle(train_idx)
            rng.shuffle(val_idx)
            rng.shuffle(test_idx)
            return train_idx, val_idx, test_idx

        if labels is None:
            all_idx = list(range(n_total))
            rng = random.Random(rand_seed)
            rng.shuffle(all_idx)
            train_size = int(0.8 * n_total)
            val_size = int(0.1 * n_total)
            train_idx = all_idx[:train_size]
            val_idx = all_idx[train_size: train_size + val_size]
            test_idx = all_idx[train_size + val_size:]
        else:
            train_idx, val_idx, test_idx = _stratified_indices(
                labels, rand_seed)

        # Persist class balance for Stage 1/2 loss weighting.
        self._split_stats = {
            "train": {"pos": 0, "neg": 0},
            "val": {"pos": 0, "neg": 0},
            "test": {"pos": 0, "neg": 0},
        }
        if labels is not None:
            def _count(idxs):
                pos = sum(1 for i in idxs if int(labels[i]) == 1)
                neg = len(idxs) - pos
                return pos, neg

            tr_pos, tr_neg = _count(train_idx)
            va_pos, va_neg = _count(val_idx)
            te_pos, te_neg = _count(test_idx)
            self._split_stats["train"] = {"pos": tr_pos, "neg": tr_neg}
            self._split_stats["val"] = {"pos": va_pos, "neg": va_neg}
            self._split_stats["test"] = {"pos": te_pos, "neg": te_neg}
            try:
                self.logger.info(
                    f"[SPLIT DEBUG] Train={len(train_idx)} (pos={tr_pos}, neg={tr_neg}) | "
                    f"Val={len(val_idx)} (pos={va_pos}, neg={va_neg}) | "
                    f"Test={len(test_idx)} (pos={te_pos}, neg={te_neg})"
                )
            except Exception:
                pass

        train_ds = Subset(self.dataset, train_idx)
        val_ds = Subset(self.dataset, val_idx)
        test_ds = Subset(self.dataset, test_idx)

        # Create Loaders
        # IMPORTANT: custom_collate needs to know the stage!
        def collate_fn(b): return custom_collate(b, stage=self.stage)

        self.train_loader = GraphDataLoader(
            train_ds, batch_size=self.batch_size, shuffle=True, collate_fn=collate_fn)
        self.val_loader = GraphDataLoader(
            val_ds, batch_size=self.batch_size, shuffle=False, collate_fn=collate_fn)
        self.test_loader = GraphDataLoader(
            test_ds, batch_size=self.batch_size, shuffle=False, collate_fn=collate_fn)

    def setup_model(self):
        # 1. Infer Dimensions from a sample
        sample = None
        for idx in range(len(self.dataset)):
            sample = self.dataset[idx]
            if sample is not None:
                break
        if sample is None:
            raise ValueError(
                "No valid sample found in dataset for model setup.")

        # Get rel_names from processor
        self.rel_names = self.dataset.CPG_Proccessor.rel_names

        # Use embedding_dims from CPG_Processor
        node_dims = {
            'cfg_node': self.embedding_dims['cfg_node'],
            'ast_node': self.embedding_dims['ast_node'],
        }

        edge_dims = {}
        rel_names = []
        if self.stage == 3:
            g = sample['graph']
            # Use processor-wide rel_names so all standardized graphs are supported.
            rel_names = [_et for _et in (self.rel_names or [])]
            if not rel_names:
                rel_names = g.canonical_etypes
            edge_dims = {et: self.embedding_dims['edge'] for et in rel_names}

        # 2. Define Output Dim
        # Stage 3: 8 Classes (Multi-label)
        # Stage 1/2: 1 Class (Binary)
        out_dim = 8 if self.stage == 3 else 1

        # 3. Instantiate Model
        try:
            from experiments.models.baseline_6 import CascadedHeteroModel
            from experiments.models.baseline_simple import SimpleHeteroModel
        except Exception:
            from models.baseline_6 import CascadedHeteroModel
            from models.baseline_simple import SimpleHeteroModel
            
        self.model = CascadedHeteroModel(
            node_dims=node_dims,
            edge_dims=edge_dims,
            hidden_dim=self.hidden_dim,
            out_dim=out_dim,
            rel_names=rel_names,
            stage=self.stage
        ).to(self.device)

        # 4. Optimizer & Loss & Scheduler
        base_optimizer = optim.AdamW(
            self.model.parameters(), lr=self.learning_rate, weight_decay=0.01)
        self.optimizer = PCGrad(base_optimizer)
        
        # Add warmup + cosine scheduler for Stage 1/2
        if self.stage in [1, 2]:
            total_steps = len(self.train_loader) * self.num_epochs
            warmup_steps = int(0.1 * total_steps)  # 10% warmup
            self.scheduler = optim.lr_scheduler.OneCycleLR(
                base_optimizer,
                max_lr=self.learning_rate * 3,  # Peak at 3x base LR
                total_steps=total_steps,
                pct_start=0.1,  # 10% warmup
                anneal_strategy='cos'
            )
            self.logger.info(f"[SCHEDULER] OneCycleLR: max_lr={self.learning_rate * 3:.2e}, warmup={warmup_steps} steps")
            self.reduce_on_plateau = None
        else:
            # For Stage 3, use ReduceLROnPlateau to handle overfitting
            self.scheduler = None
            self.reduce_on_plateau = optim.lr_scheduler.ReduceLROnPlateau(
                base_optimizer,
                mode='min',
                factor=0.5,
                patience=3,
                verbose=True,
                min_lr=1e-8
            )
            self.logger.info(f"[SCHEDULER] ReduceLROnPlateau: factor=0.5, patience=3")

        # Stage-specific loss:
        # - Stage 1/2: FocalLoss for extreme imbalance (better than BCE for rare positives)
        # - Stage 3: FocalLoss (multilabel) with per-class weights
        if self.stage in [1, 2]:
            pos = int(getattr(self, "_split_stats", {}).get(
                "train", {}).get("pos", 0))
            neg = int(getattr(self, "_split_stats", {}).get(
                "train", {}).get("neg", 0))
            
            # Use FocalLoss with high alpha for rare positives
            if pos > 0 and neg > 0:
                pos_weight = torch.tensor(
                    [neg / max(1, pos)], dtype=torch.float32, device=self.device)
                self.logger.info(
                    f"[LOSS DEBUG] Stage{self.stage} BCE pos_weight={float(pos_weight.item()):.4f}")
                self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
            else:
                self.logger.warning(
                    f"[LOSS DEBUG] Stage{self.stage} has pos={pos}, neg={neg} in train; using unweighted BCEWithLogitsLoss"
                )
                self.criterion = nn.BCEWithLogitsLoss()
        else:
            # Compute per-class weights for Stage 3 multilabel
            pos_weight = self._compute_stage3_class_weights()
            self.criterion = FocalLoss(alpha=.65, gamma=2.5, pos_weight=pos_weight, reduction="none")
            if pos_weight is not None:
                self.logger.info(f"[LOSS DEBUG] Per-class weights: {pos_weight.cpu().numpy()}")
        
        # Store best thresholds for Stage 3 (initialized to 0.5)
        self.best_thresholds = torch.full((8,), 0.5, dtype=torch.float32) if self.stage == 3 else None

    def _compute_stage3_class_weights(self):
        """Compute per-class weights for Stage 3 multilabel classification."""
        if self.stage != 3:
            return None
        
        self.logger.info("Computing per-class weights for Stage 3...")
        class_pos_counts = torch.zeros(8, dtype=torch.float32)
        class_neg_counts = torch.zeros(8, dtype=torch.float32)
        
        # Count positive/negative samples per class across training data
        for batch in self.train_loader:
            if batch is None:
                continue
            
            cfg_labels = batch.get('cfg_labels')
            ast_labels = batch.get('ast_labels')
            
            if cfg_labels is not None:
                class_pos_counts += cfg_labels.sum(dim=0).cpu()
                class_neg_counts += (1 - cfg_labels).sum(dim=0).cpu()
            
            if ast_labels is not None:
                class_pos_counts += ast_labels.sum(dim=0).cpu()
                class_neg_counts += (1 - ast_labels).sum(dim=0).cpu()
        
        # Compute pos_weight = neg_count / max(pos_count, 1) for each class
        pos_weight = class_neg_counts / torch.clamp(class_pos_counts, min=1.0)
        
        # Cap maximum weight to prevent extreme values
        pos_weight = torch.clamp(pos_weight, min=1.0, max=3.0)
        
        self.logger.info(f"Class positive counts: {class_pos_counts.numpy()}")
        self.logger.info(f"Class negative counts: {class_neg_counts.numpy()}")
        
        return pos_weight.to(self.device)
    
    def _prepare_batch(self, batch):
        """Moves batch to device and extracts labels based on stage."""
        if self.is_graph_level:
            g = batch['graph'].to(self.device)

            # Labels are dicts: {"ContractA": 0, "ContractB": 1, ...}
            labels_raw = batch['graph_labels']  # List of dicts
            project_names = batch['project_names']

            # Build a mapping of (batch_idx, entity_name) -> label
            label_map = {}
            for batch_idx, (p_name, lbl_data) in enumerate(zip(project_names, labels_raw)):
                if isinstance(lbl_data, dict):
                    for entity_name, label_val in lbl_data.items():
                        # Store with batch index for matching
                        label_map[(batch_idx, entity_name)] = float(label_val)

            return g, label_map

        else:
            # Stage 3: Labels are Tensors [Total_Nodes, 8]
            g = batch['graph'].to(self.device)
            targets = {
                'cfg': batch['cfg_labels'].to(self.device) if batch['cfg_labels'] is not None else None,
                'ast': batch['ast_labels'].to(self.device) if batch['ast_labels'] is not None else None
            }
            return g, targets

    def train(self):
        best_f1 = 0.0
        patience_counter = 0

        for epoch in range(self.num_epochs):
            self.model.train()
            epoch_loss = 0

            batch_count = 0
            for batch in tqdm(self.train_loader, desc=f"Ep {epoch+1} Train"):
                if batch is None:
                    continue

                inputs, label_data = self._prepare_batch(batch)

                self.optimizer.zero_grad()
                preds = self.model({'graph': inputs})

                loss = 0
                if self.is_graph_level:
                    loss = self._train_step_graph(preds, label_data)
                else:
                    loss = self._train_step_node(preds, label_data)

                if loss == 0:
                    continue
                # Use PCGrad for backward
                if isinstance(loss, list):
                    self.optimizer.pc_backward(loss)
                    epoch_loss += sum(loss_item.item() for loss_item in loss)
                else:
                    self.optimizer.pc_backward([loss])
                    epoch_loss += loss.item()
                
                # Gradient clipping for stability
                grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                # CRITICAL: Actually update the weights!
                self.optimizer.step()
                
                batch_count += 1
                # Log first batch to verify training
                if batch_count == 1:
                    if isinstance(loss, list):
                        self.logger.info(f"[TRAIN DEBUG] Epoch {epoch+1} Batch 1 Loss: {[l.item() for l in loss]} | Grad Norm: {grad_norm:.4f}")
                    else:
                        self.logger.info(f"[TRAIN DEBUG] Epoch {epoch+1} Batch 1 Loss: {loss.item():.4f} | Grad Norm: {grad_norm:.4f}")
                
                # Step scheduler if using OneCycleLR
                if self.scheduler is not None:
                    self.scheduler.step()

            # Validation
            val_metrics = self.evaluate(self.val_loader)
            avg_loss = epoch_loss / max(len(self.train_loader), 1)
            self.logger.info(
                f"Epoch {epoch+1}: Loss={avg_loss:.4f} | Val F1={val_metrics['f1']:.4f} | Val AUC={val_metrics['auc']:.4f}")

            # Step scheduler if using ReduceLROnPlateau (Stage 3)
            if self.reduce_on_plateau is not None:
                self.reduce_on_plateau.step(val_metrics['loss'])

            # Update history
            self.history["train_loss"].append(
                epoch_loss / len(self.train_loader))
            self.history["val_loss"].append(val_metrics['loss'])
            self.history["val_f1"].append(val_metrics['f1'])
            self.history["val_auc"].append(val_metrics['auc'])

            # Checkpoint
            if val_metrics['f1'] > best_f1:
                best_f1 = val_metrics['f1']
                patience_counter = 0
                torch.save(self.model.state_dict(),
                           f"{self.log_folder}/best_model.pth")
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    self.logger.info("Early Stopping")
                    break

    def _train_step_graph(self, preds, label_map):
        """Training step for stages 1-2: Graph-level classification with one entity per graph."""
        # preds['graph_logits']: [batch_size, 1]
        # label_map: {(batch_idx, entity_name): label} - one per graph

        # Sort by batch_idx to get targets in order
        sorted_keys = sorted(label_map.keys(), key=lambda x: x[0])
        targets = torch.tensor(
            [label_map[k] for k in sorted_keys], dtype=torch.float32).to(self.device)

        logits = preds['graph_logits'].squeeze(-1)  # [batch_size]

        # Support either FocalLoss (legacy) or BCEWithLogitsLoss (preferred for Stage 1/2).
        if isinstance(self.criterion, FocalLoss):
            loss = self.criterion(
                logits.unsqueeze(-1), targets.unsqueeze(-1)
            ).mean()
        else:
            loss = self.criterion(logits, targets)
        return loss

    def _train_step_node(self, preds, label_data):
        """Training step for stage 3: Node-level multilabel classification."""
        targets = label_data
        losses = []
        
        # Apply label smoothing to prevent overconfidence
        label_smoothing = 0.05
        
        if targets['cfg'] is not None:
            smoothed_cfg = targets['cfg'] * (1 - label_smoothing) + 0.5 * label_smoothing
            losses.append(self.criterion(
                preds['cfg_logits'], smoothed_cfg).mean())
        if targets['ast'] is not None:
            smoothed_ast = targets['ast'] * (1 - label_smoothing) + 0.5 * label_smoothing
            # Balance AST and CFG losses equally (not 0.5x)
            losses.append(
                self.criterion(preds['ast_logits'], smoothed_ast).mean())
        return losses if losses else 0

    def evaluate(self, loader, fixed_thresholds=None, optimize_thresholds=True):
        self.model.eval()
        all_preds = []
        all_targets = []
        epoch_loss = 0

        with torch.no_grad():
            for batch in loader:
                if batch is None:
                    continue
                inputs, label_data = self._prepare_batch(batch)
                preds = self.model({'graph': inputs})

                # Compute loss
                if self.is_graph_level:
                    loss = self._train_step_graph(preds, label_data)
                else:
                    losses = self._train_step_node(preds, label_data)
                    loss = sum(losses) if isinstance(losses, list) else losses
                epoch_loss += loss.item() if loss != 0 else 0

                if self.is_graph_level:
                    probs, targets = self._eval_step_graph(preds, label_data)
                else:
                    probs, targets = self._eval_step_node(preds, label_data)

                if probs is None or targets is None:
                    continue

                all_preds.append(probs)
                all_targets.append(targets)

        if not all_preds:
            return {'f1': 0, 'auc': 0, 'loss': 0, 'y_pred': np.array([]), 'y_true': np.array([])}

        y_prob = torch.cat(all_preds).numpy()
        y_true = torch.cat(all_targets).numpy()

        # Debug: Log label distribution
        pos_count = y_true.sum()
        neg_count = len(y_true) - pos_count
        self.logger.info(
            f"[EVAL DEBUG] Samples: {len(y_true)} | Positive: {pos_count} ({pos_count/len(y_true)*100:.1f}%) | Negative: {neg_count}")

        # Metrics
        thr = 0.5
        thresholds_used = None
        if self.stage in [1, 2]:
            # Tune threshold for macro-F1 on this eval split (skips if single-class).
            try:
                unique = np.unique(y_true)
                if unique.size >= 2:
                    best_f1 = -1.0
                    best_thr = 0.5
                    for t in np.linspace(0.05, 0.95, 91):
                        yp = (y_prob > t).astype(int)
                        f1_t = f1_score(
                            y_true, yp, average='macro', zero_division=0)
                        if f1_t > best_f1:
                            best_f1 = f1_t
                            best_thr = float(t)
                    thr = best_thr
                    self.logger.info(
                        f"[EVAL DEBUG] Best threshold={thr:.3f} (macro-F1={best_f1:.4f})")
            except Exception as e:
                self.logger.warning(
                    f"[EVAL DEBUG] Threshold tuning failed: {e}")

            y_pred = (y_prob > thr).astype(int)
        else:
            # Stage 3: per-class thresholds
            num_classes = y_prob.shape[1]
            if fixed_thresholds is not None:
                thresholds_used = np.array(fixed_thresholds, dtype=float)
                self.logger.info(f"[EVAL DEBUG] Using fixed thresholds: {thresholds_used}")
            elif optimize_thresholds:
                thresholds_used = np.zeros(num_classes)
                for class_idx in range(num_classes):
                    y_true_class = y_true[:, class_idx]
                    y_prob_class = y_prob[:, class_idx]
                    if y_true_class.sum() == 0:
                        thresholds_used[class_idx] = 0.9  # conservative when no positives
                        continue
                    best_score = -1.0
                    best_thr = 0.5
                    for t in np.linspace(0.25, 0.8, 12):  # conservative, slightly wider
                        yp = (y_prob_class >= t).astype(int)
                        if yp.sum() == 0:
                            continue
                        f1_t = f1_score(y_true_class, yp, average='binary', zero_division=0)
                        prec_t = precision_score(y_true_class, yp, zero_division=0)
                        score = 0.3 * f1_t + 0.7 * prec_t  # emphasize precision to curb FPs
                        if score > best_score:
                            best_score = score
                            best_thr = float(t)
                    thresholds_used[class_idx] = max(best_thr, 0.4)
                self.logger.info(f"[EVAL DEBUG] Per-class thresholds (optimized): {thresholds_used}")
                self.best_thresholds = torch.tensor(thresholds_used, dtype=torch.float32)
            else:
                thresholds_used = np.full(num_classes, 0.5)

            thresholds_used = np.array(thresholds_used, dtype=float)
            y_pred = (y_prob >= thresholds_used).astype(int)

        # Debug: Log prediction distribution
        pred_pos = y_pred.sum()
        pred_neg = len(y_pred) - pred_pos
        self.logger.info(
            f"[EVAL DEBUG] Predictions: Positive: {pred_pos} ({pred_pos/len(y_pred)*100:.1f}%) | Negative: {pred_neg}")
        self.logger.info(
            f"[EVAL DEBUG] Prob range: [{y_prob.min():.4f}, {y_prob.max():.4f}] | Mean: {y_prob.mean():.4f}")

        # Macro F1 is safer for imbalance
        f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)

        try:
            auc = roc_auc_score(y_true, y_prob, average='macro')
        except Exception:
            auc = 0.5  # Fail gracefully if only one class present

        # Classification report
        if self.stage == 3:
            # Per-label report for multilabel Stage 3
            report = classification_report(y_true, y_pred, target_names=[
                                           f'Vuln{i}' for i in range(8)], zero_division=0)
            self.logger.info(
                f"Per-Label Classification Report (Stage 3):\n{report}")
        else:
            # Binary report for Stages 1/2
            report = classification_report(y_true, y_pred, target_names=[
                                           'Safe', 'Vuln'], zero_division=0)
            self.logger.info(f"Classification Report:\n{report}")

        return {
            'f1': f1,
            'auc': auc,
            'loss': epoch_loss / len(loader),
            'y_pred': y_pred,
            'y_true': y_true,
            'thresholds': thresholds_used,
        }

    def _eval_step_graph(self, preds, label_map):
        """Evaluation step for stages 1-2: Graph-level classification."""
        sorted_keys = sorted(label_map.keys(), key=lambda x: x[0])
        targets = torch.tensor(
            [label_map[k] for k in sorted_keys], dtype=torch.float32).to(self.device)

        probs = torch.sigmoid(
            preds['graph_logits'].squeeze(-1))  # [batch_size]
        return probs.cpu(), targets.cpu()

    def _eval_step_node(self, preds, label_data):
        """Evaluation step for stage 3: Node-level classification."""
        targets = label_data
        if targets['cfg'] is None and targets['ast'] is None:
            return None, None

        all_probs = []
        all_targets = []

        if targets['cfg'] is not None:
            all_probs.append(torch.sigmoid(preds['cfg_logits']))
            all_targets.append(targets['cfg'])

        if targets['ast'] is not None:
            all_probs.append(torch.sigmoid(preds['ast_logits']))
            all_targets.append(targets['ast'])

        if all_probs:
            combined_probs = torch.cat(all_probs, dim=0)
            combined_targets = torch.cat(all_targets, dim=0)
            return combined_probs.cpu(), combined_targets.cpu()

        return None, None

    def test(self):
        """Evaluate on test set."""
        return self.evaluate(self.test_loader)
