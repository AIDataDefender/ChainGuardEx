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
from torch.utils.data import random_split
from sklearn.metrics import (
    roc_auc_score,
    f1_score,
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

        # Clamp probabilities to avoid log(0)
        eps = 1e-6
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
        learning_rate=3e-5,
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

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self.num_epochs = num_epochs
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
        # Basic Split (80/10/10)
        # Note: For Stage 1/2, stratification is tricky without extracting all labels first
        # We use random split for simplicity in this updated version
        train_size = int(0.8 * len(self.dataset))
        val_size = int(0.1 * len(self.dataset))
        test_size = len(self.dataset) - train_size - val_size

        train_ds, val_ds, test_ds = random_split(
            self.dataset, [train_size, val_size, test_size],
            generator=torch.Generator().manual_seed(rand_seed)
        )

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
            raise ValueError("No valid sample found in dataset for model setup.")

        # Get rel_names from processor
        self.rel_names = self.dataset.CPG_Proccessor.rel_names

        # Use embedding_dims from CPG_Processor
        node_dims = {
            'cfg_node': self.embedding_dims['node_cfg'],
            'ast_node': self.embedding_dims['node_ast'],
        }

        edge_dims = {}
        rel_names = []
        if self.stage == 3:
            g = sample['graph']
            edge_dims = {et: self.embedding_dims['edge']
                        for et in g.canonical_etypes}
            rel_names = g.canonical_etypes

        # 2. Define Output Dim
        # Stage 3: 8 Classes (Multi-label)
        # Stage 1/2: 1 Class (Binary)
        out_dim = 8 if self.stage == 3 else 1

        # 3. Instantiate Model
        from models.baseline_5 import CascadedHeteroModel
        self.model = CascadedHeteroModel(
            node_dims=node_dims,
            edge_dims=edge_dims,
            hidden_dim=self.hidden_dim,
            out_dim=out_dim,
            rel_names=rel_names,
            stage=self.stage
        ).to(self.device)

        # 4. Optimizer & Loss
        self.optimizer = PCGrad(optim.AdamW(
            self.model.parameters(), lr=self.learning_rate))

        # Use FocalLoss for both graph-level and node-level to handle class imbalance
        # FocalLoss automatically handles imbalance by focusing on hard examples
        # Change to "none" for PCGrad
        self.criterion = FocalLoss(alpha=0.25, gamma=2.0, reduction="none")

    def _prepare_batch(self, batch):
        """Moves batch to device and extracts labels based on stage."""
        if self.is_graph_level:
            # Stage 1/2: Embeddings are pre-computed [batch_size, emb_dim]
            embeddings = batch['embeddings'].to(self.device)

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

            # Return embeddings and label mapping
            return embeddings, label_map

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

            for batch in tqdm(self.train_loader, desc=f"Ep {epoch+1} Train"):
                if batch is None:
                    continue

                inputs, label_data = self._prepare_batch(batch)

                self.optimizer.zero_grad()
                preds = self.model({'graph': inputs} if not self.is_graph_level else {
                                   'embeddings': inputs})

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
                    epoch_loss += sum(l.item() for l in loss)
                else:
                    self.optimizer.pc_backward([loss])
                    epoch_loss += loss.item()

            # Validation
            val_metrics = self.evaluate(self.val_loader)
            self.logger.info(
                f"Epoch {epoch+1}: Loss={epoch_loss:.4f} | Val F1={val_metrics['f1']:.4f} | Val AUC={val_metrics['auc']:.4f}")

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

        # FocalLoss expects [N, C], C=1
        loss = self.criterion(logits.unsqueeze(-1),
                              targets.unsqueeze(-1)).mean()
        return loss

    def _train_step_node(self, preds, label_data):
        """Training step for stage 3: Node-level multilabel classification."""
        targets = label_data
        losses = []
        if targets['cfg'] is not None:
            losses.append(self.criterion(
                preds['cfg_logits'], targets['cfg']).mean())
        if targets['ast'] is not None:
            losses.append(
                0.5 * self.criterion(preds['ast_logits'], targets['ast']).mean())
        return losses if losses else 0

    def evaluate(self, loader):
        self.model.eval()
        all_preds = []
        all_targets = []
        epoch_loss = 0

        with torch.no_grad():
            for batch in loader:
                if batch is None:
                    continue
                inputs, label_data = self._prepare_batch(batch)
                preds = self.model({'graph': inputs} if not self.is_graph_level else {
                                   'embeddings': inputs})

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
        y_pred = (y_prob > 0.5).astype(int)

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
        except:
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

        return {'f1': f1, 'auc': auc, 'loss': epoch_loss / len(loader), 'y_pred': y_pred, 'y_true': y_true}

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
