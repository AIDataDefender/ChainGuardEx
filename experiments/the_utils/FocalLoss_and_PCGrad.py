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

    def zero_grad(self, set_to_none: bool = False):
        # Keep signature compatible with torch.optim.Optimizer.zero_grad
        # so callers can use the faster set_to_none path.
        try:
            return self._optim.zero_grad(set_to_none=set_to_none)
        except TypeError:
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
            length = int(np.prod(shape))
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

