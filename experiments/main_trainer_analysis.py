import os
import time
import traceback
import argparse
import json
import math
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc, multilabel_confusion_matrix
import numpy as np
from sklearn.metrics import roc_auc_score
from base_trainer import BaseTrainer
import torch
import dgl
from torchinfo import summary  # type: ignore
try:
    from experiments.the_utils.graph_utils import OWASP_VULN
except ImportError:
    from the_utils.graph_utils import OWASP_VULN


class TrainerWithAnalysis(BaseTrainer):
    """
    Extended trainer that adapts analysis and visualization based on the 
    Stage (Graph vs Node Classification).
    """

    def __init__(self, do_test_data=True, **kwargs):
        self.do_test_data = do_test_data
        super().__init__(**kwargs)

        # Store training start time for efficiency tracking
        self.start_time = time.time()
        self.training_time = 0

    def run_pre_train_analysis(self):
        """Run comprehensive checks before training starts."""
        if self.do_test_data:
            self.test_data_loader()
        self._log_model_summary()

    def test_data_loader(self):
        """Analyze batch composition for Train/Val/Test."""
        self.logger.info("=" * 40)
        self.logger.info(f"Testing DataLoader (Stage {self.stage})")
        self.logger.info("=" * 40)

        for name, loader in [("TRAIN", self.train_loader), ("VAL", self.val_loader)]:
            self.logger.info(f"\n# {name} DATASET")
            self._analyze_dataloader(loader)

    def _analyze_dataloader(self, loader):
        """Robust analysis handling both Tensor labels (S3) and Dict labels (S1/2)."""
        try:
            pos_samples = 0
            total_samples = 0
            class_counts = None
            if self.stage == 3:
                class_counts = torch.zeros(8, dtype=torch.float32)

            def _label_to_int(v):
                """Coerce various label representations to {0,1}."""
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

            # Check first batch only for brevity
            for batch_idx, batch in enumerate(loader):
                if batch is None:
                    continue

                g = batch['graph']
                batch_size = g.batch_size

                if batch_idx == 0:
                    self.logger.info(f"  Batch 0 Stats:")  # noqa: F541
                    self.logger.info(f"    Graphs: {batch_size}")
                    self.logger.info(
                        f"    Nodes: {g.num_nodes()} (Avg {g.num_nodes()/batch_size:.1f})")
                    self.logger.info(f"    Edges: {g.num_edges()}")
                    self.logger.info(f"    Device: {g.device}")

                # 2. Label Stats
                # Extract logic similar to BaseTrainer._prepare_batch
                if self.stage in [1, 2]:
                    # Graph Level: Label is in batch['graph_labels'] as list/dict
                    raw = batch['graph_labels']

                    # DEBUG: Log first batch labels to understand structure
                    if batch_idx == 0:
                        self.logger.info(
                            "  DEBUG - Label dicts for first 3 GRAPHS (each graph is one contracts):")
                        for i, graph_labels in enumerate(raw[:3] if len(raw) >= 3 else raw):
                            num_contracts = len(graph_labels) if isinstance(
                                graph_labels, dict) else 1
                            num_vuln = (
                                sum(_label_to_int(v)
                                    for v in graph_labels.values())
                                if isinstance(graph_labels, dict)
                                else _label_to_int(graph_labels)
                            )
                            self.logger.info(
                                f"    Graph {i+1}: {num_contracts} contracts, {num_vuln} vulnerable")
                            if isinstance(graph_labels, dict):
                                vuln_contracts = [
                                    k for k, v in graph_labels.items() if _label_to_int(v) == 1]
                                if vuln_contracts:
                                    self.logger.info(
                                        f"      Vulnerable: {vuln_contracts}")

                    # Convert to binary list
                    curr_pos = 0
                    for item in raw:
                        if isinstance(item, dict):
                            # Count a graph as positive if any entity within it is vulnerable.
                            if any(_label_to_int(v) == 1 for v in item.values()):
                                curr_pos += 1
                        else:
                            curr_pos += _label_to_int(item)

                    pos_samples += curr_pos
                    total_samples += len(raw)

                else:
                    # Node Level: Label is Tensor [N, 8]
                    cfg_lbls = batch.get('cfg_labels')
                    ast_lbls = batch.get('ast_labels')

                    # Count vulnerable nodes across both views
                    if cfg_lbls is not None:
                        pos_samples += (cfg_lbls.sum(dim=1) > 0).sum().item()
                        total_samples += cfg_lbls.shape[0]
                        if class_counts is not None:
                            class_counts += cfg_lbls.sum(dim=0).cpu()

                    if ast_lbls is not None:
                        pos_samples += (ast_lbls.sum(dim=1) > 0).sum().item()
                        total_samples += ast_lbls.shape[0]
                        if class_counts is not None:
                            class_counts += ast_lbls.sum(dim=0).cpu()

            # Summary
            if total_samples > 0:
                pos_pct = (pos_samples / total_samples) * 100
                entity = "GRAPHS" if self.stage in [1, 2] else "NODES"
                self.logger.info(f"  Total {entity}: {total_samples}")
                self.logger.info(
                    f"  Vulnerable {entity}: {pos_samples} ({pos_pct:.2f}%)")
                if self.stage == 3 and class_counts is not None:
                    self.logger.info(
                        f"  Per-Class Positive Counts: {class_counts.numpy()}")
                    self.logger.info(
                        f"  Class Imbalance Ratios: {(class_counts / class_counts.sum()).numpy()}")

        except Exception as e:
            self.logger.error(f"Analysis failed: {e}")
            traceback.print_exc()

    def _log_model_summary(self):
        """Generate and log model summary."""
        try:

            self.logger.info("Generating model summary...")

            summary_batch = next(iter(self.val_loader))

            # Move the sample batch to the device
            batch_gpu = {}
            for key, tensor in summary_batch.items():
                if isinstance(tensor, torch.Tensor):
                    batch_gpu[key] = tensor.to(self.device)
                elif isinstance(tensor, dgl.DGLGraph):
                    batch_gpu[key] = tensor.to(self.device)

            self.logger.info("" + "=" * 40)
            self.logger.info("--- Model Summary ---")

            # Generate the summary by passing the sample batch as input_data
            summary(self.model,
                    input_data=batch_gpu,
                    depth=8,
                    col_names=["input_size", "output_size", "num_params", "mult_adds"])

            self.logger.info("=" * 40 + "\n")

        except Exception as e:
            self.logger.warning(f"Could not generate model summary: {e}")
            self.logger.info(
                f"Model Architecture (simple):\n{self.model}")  # Fallback

    def save_history_plot(self):
        """Save training history plot."""
        try:
            os.makedirs(f"{self.log_folder}/viz", exist_ok=True)

            if self.history["train_loss"]:
                fig, axes = plt.subplots(1, 2, figsize=(14, 6))

                # Loss
                axes[0].plot(self.history["train_loss"], label="Train")
                axes[0].plot(self.history["val_loss"], label="Val")
                axes[0].set_title(f"Stage {self.stage} Loss")
                axes[0].set_xlabel("Epoch")
                axes[0].set_ylabel("Loss")
                axes[0].legend()

                # Metrics
                axes[1].plot(self.history["val_f1"],
                             label="Macro F1", color="green")
                axes[1].plot(self.history["val_auc"],
                             label="AUC", color="purple")
                if self.stage == 3:
                    axes[1].plot(self.history["val_hamming"],
                                 label="Hamming Score", color="orange")
                axes[1].set_title("Validation Metrics")
                axes[1].set_xlabel("Epoch")
                axes[1].legend()

                plt.savefig(
                    f"{self.log_folder}/viz/history_stage{self.stage}.png")
                plt.close()

        except Exception as e:
            self.logger.error(f"History plot save failed: {e}")
            traceback.print_exc()

    def visualize_results(self, test_metrics=None):
        """Generate Stage-specific visualizations."""
        try:
            os.makedirs(f"{self.log_folder}/viz", exist_ok=True)

            # 1. Training History
            self.save_history_plot()
            if test_metrics and "y_pred" in test_metrics:
                y_true = np.asarray(test_metrics.get("y_true"))
                y_pred = np.asarray(test_metrics.get("y_pred"))
            else:
                return

            # Guard: nothing to visualize
            if y_true.size == 0 or y_pred.size == 0:
                self.logger.warning(
                    "[visualize_results] Empty y_true/y_pred; skipping visualizations.")
                return

            is_multilabel = (y_true.ndim == 2 and y_true.shape[1] > 1)

            if self.stage == 3 and is_multilabel:
                # Combined multilabel confusion matrix (Safe + all OWASP labels).
                # Rows: true labels, Cols: predicted labels. A node can contribute to multiple cells.
                num_classes = int(y_true.shape[1])
                label_names = ["Safe"] + [
                    (OWASP_VULN[i] if i < len(OWASP_VULN) else f"Class {i}")
                    for i in range(num_classes)
                ]
                cm = np.zeros((num_classes + 1, num_classes + 1),
                              dtype=np.int64)

                for i in range(y_true.shape[0]):
                    true_idx = np.flatnonzero(y_true[i].astype(int))
                    pred_idx = np.flatnonzero(y_pred[i].astype(int))

                    # Use index 0 to represent Safe (no labels).
                    if true_idx.size == 0:
                        true_bins = np.array([0], dtype=int)
                    else:
                        true_bins = true_idx + 1

                    if pred_idx.size == 0:
                        pred_bins = np.array([0], dtype=int)
                    else:
                        pred_bins = pred_idx + 1

                    for t in true_bins:
                        for p in pred_bins:
                            cm[t, p] += 1

                fig, ax = plt.subplots(figsize=(12, 10))
                # Normalize per row for coloring
                cm_norm = cm.astype(float)
                row_sums = cm.sum(axis=1, keepdims=True)
                row_sums[row_sums == 0] = 1  # avoid div by zero
                cm_norm = cm / row_sums
                im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
                # Annotate with counts
                for i in range(cm.shape[0]):
                    for j in range(cm.shape[1]):
                        ax.text(j, i, f'{cm[i, j]}', ha='center', va='center', 
                                color='black' if cm_norm[i, j] < 0.5 else 'white', fontsize=12)
                ax.set_xticks(range(len(label_names)))
                ax.set_yticks(range(len(label_names)))
                ax.set_xticklabels(label_names, rotation=45, ha='right')
                ax.set_yticklabels(label_names)
                ax.set_title(f"Stage {self.stage} Multilabel Confusion Matrix")
                plt.colorbar(im, ax=ax)
                plt.savefig(f"{self.log_folder}/viz/cm_stage{self.stage}.png")
                plt.close()


                # Second CM: only vuln labels (no Safe)
                num_classes = int(y_true.shape[1])
                label_names_vuln = [
                    (OWASP_VULN[i] if i < len(OWASP_VULN) else f"Class {i}")
                    for i in range(num_classes)
                ]
                cm_vuln = np.zeros((num_classes, num_classes), dtype=np.int64)
                for i in range(y_true.shape[0]):
                    true_idx = np.flatnonzero(y_true[i].astype(int))
                    pred_idx = np.flatnonzero(y_pred[i].astype(int))
                    for t in true_idx:
                        for p in pred_idx:
                            cm_vuln[t, p] += 1
                fig, ax = plt.subplots(figsize=(12, 10))
                # Normalize per row for coloring
                cm_vuln_norm = cm_vuln.astype(float)
                row_sums = cm_vuln.sum(axis=1, keepdims=True)
                row_sums[row_sums == 0] = 1  # avoid div by zero
                cm_vuln_norm = cm_vuln / row_sums
                im = ax.imshow(cm_vuln_norm, cmap="Blues", vmin=0, vmax=1)
                # Annotate with counts
                for i in range(cm_vuln.shape[0]):
                    for j in range(cm_vuln.shape[1]):
                        ax.text(j, i, f'{cm_vuln[i, j]}', ha='center', va='center', 
                                color='black' if cm_vuln_norm[i, j] < 0.5 else 'white', fontsize=12)
                ax.set_xticks(range(len(label_names_vuln)))
                ax.set_yticks(range(len(label_names_vuln)))
                ax.set_xticklabels(label_names_vuln, rotation=45, ha='right')
                ax.set_yticklabels(label_names_vuln)
                ax.set_title(f"Stage {self.stage} Multilabel Confusion Matrix (Vuln Only)")
                plt.colorbar(im, ax=ax)
                plt.tight_layout()
                plt.savefig(f"{self.log_folder}/viz/cm_vuln_only_stage{self.stage}.png")
                plt.close()

                # Third: sklearn multilabel_confusion_matrix
                mcm = multilabel_confusion_matrix(y_true, y_pred)
                # Plot each class's binary cm in a grid
                n_classes = y_true.shape[1]
                n_cols = 4  # adjust as needed
                n_rows = math.ceil(n_classes / n_cols)
                fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols*5, n_rows*4))
                if n_rows == 1:
                    axes = axes.reshape(1, -1)
                for i in range(n_classes):
                    cm = mcm[i]
                    row = i // n_cols
                    col = i % n_cols
                    ax = axes[row, col]
                    im = ax.imshow(cm, cmap="Blues")
                    # Annotate with counts
                    for ii in range(cm.shape[0]):
                        for jj in range(cm.shape[1]):
                            ax.text(jj, ii, f'{cm[ii, jj]}', ha='center', va='center', 
                                    color='black' if cm[ii, jj] < cm.max() / 2 else 'white', fontsize=12)
                    ax.set_xticks(range(2))
                    ax.set_yticks(range(2))
                    ax.set_xticklabels(["Not " + label_names_vuln[i], label_names_vuln[i]], rotation=45, ha='right')
                    ax.set_yticklabels(["Not " + label_names_vuln[i], label_names_vuln[i]])
                    ax.set_title(f"{label_names_vuln[i]}")
                # Hide unused subplots
                for i in range(n_classes, n_rows * n_cols):
                    row = i // n_cols
                    col = i % n_cols
                    axes[row, col].set_visible(False)
                plt.tight_layout()
                plt.savefig(f"{self.log_folder}/viz/mcm_stage{self.stage}.png")
                plt.close()

            else:
                # Binary Confusion Matrix
                if y_true.ndim > 1:
                    y_true_flat = (y_true.sum(axis=1) > 0).astype(int)
                    y_pred_flat = (y_pred.sum(axis=1) > 0).astype(int)
                else:
                    y_true_flat = y_true.reshape(-1)
                    y_pred_flat = y_pred.reshape(-1)
                    if y_pred_flat.dtype != np.int64 and y_pred_flat.dtype != np.int32 and y_pred_flat.dtype != np.bool_:
                        y_pred_flat = (y_pred_flat > 0.5).astype(int)
                cm = confusion_matrix(y_true_flat, y_pred_flat, labels=[0, 1])
                fig, ax = plt.subplots(figsize=(6, 6))
                im = ax.imshow(cm, cmap="Blues")
                # Annotate with counts
                for i in range(cm.shape[0]):
                    for j in range(cm.shape[1]):
                        ax.text(j, i, f'{cm[i, j]}', ha='center', va='center', 
                                color='black' if cm[i, j] < cm.max() / 2 else 'white', fontsize=12)
                ax.set_xticks(range(2))
                ax.set_yticks(range(2))
                ax.set_xticklabels(["Safe", "Vuln"], rotation=45, ha='right')
                ax.set_yticklabels(["Safe", "Vuln"])
                plt.title(f"Stage {self.stage} Confusion Matrix")
                plt.colorbar(im, ax=ax)
                plt.savefig(f"{self.log_folder}/viz/cm_stage{self.stage}.png")
                plt.close()

            # Save confusion matrix statistics
            if test_metrics and "y_pred" in test_metrics and "y_true" in test_metrics:
                cm_stats = {"stage": self.stage}
                y_true_orig = np.asarray(test_metrics["y_true"])
                y_pred_orig = np.asarray(test_metrics["y_pred"])

                if self.stage == 3 and y_true_orig.ndim == 2:
                    # Combined multilabel confusion matrix (Safe + all OWASP labels).
                    num_classes = int(y_true_orig.shape[1])
                    ml_cm = np.zeros(
                        (num_classes + 1, num_classes + 1), dtype=np.int64)
                    for i in range(y_true_orig.shape[0]):
                        true_idx = np.flatnonzero(y_true_orig[i].astype(int))
                        pred_idx = np.flatnonzero(y_pred_orig[i].astype(int))
                        true_bins = (
                            true_idx + 1) if true_idx.size > 0 else np.array([0], dtype=int)
                        pred_bins = (
                            pred_idx + 1) if pred_idx.size > 0 else np.array([0], dtype=int)
                        for t in true_bins:
                            for p in pred_bins:
                                ml_cm[t, p] += 1
                    cm_stats["multilabel_cm"] = {
                        "labels": ["Safe"] + [
                            (OWASP_VULN[i] if i < len(
                                OWASP_VULN) else f"Class {i}")
                            for i in range(num_classes)
                        ],
                        "matrix": ml_cm.tolist(),
                    }

                # Overall binary cm
                if self.stage == 3 and y_true_orig.ndim == 2:
                    y_true_flat = (y_true_orig.sum(axis=1) > 0).astype(int)
                    y_pred_flat = (y_pred_orig.sum(axis=1) > 0).astype(int)
                else:
                    y_true_flat = y_true_orig.reshape(-1)
                    y_pred_flat = y_pred_orig.reshape(-1)

                cm = confusion_matrix(y_true_flat, y_pred_flat, labels=[0, 1])
                tn, fp, fn, tp = cm.ravel()
                cm_stats["overall"] = {
                    "tp": int(tp),
                    "fp": int(fp),
                    "tn": int(tn),
                    "fn": int(fn),
                    "accuracy": float((tp + tn) / (tp + tn + fp + fn)) if tp + tn + fp + fn > 0 else 0,
                    "precision": float(tp / (tp + fp)) if tp + fp > 0 else 0,
                    "recall": float(tp / (tp + fn)) if tp + fn > 0 else 0,
                    "f1": float(2 * tp / (2 * tp + fp + fn)) if 2 * tp + fp + fn > 0 else 0,
                    "cm_matrix": cm.tolist()
                }

                with open(f"{self.log_folder}/viz/cm_stats_stage{self.stage}.json", 'w') as f:
                    json.dump(cm_stats, f, indent=4)

            # 5. ROC AUC Curves
            if test_metrics and "y_prob" in test_metrics and "y_true" in test_metrics:
                y_true = test_metrics["y_true"]
                y_true = np.asarray(y_true)
                y_prob = np.asarray(test_metrics["y_prob"])

                if self.stage in [1, 2]:
                    # Single ROC curve for binary classification
                    fpr, tpr, _ = roc_curve(y_true, y_prob)
                    roc_auc = auc(fpr, tpr)
                    plt.figure(figsize=(8, 6))
                    plt.plot(fpr, tpr, color='darkorange', lw=2,
                             label=f'ROC curve (area = {roc_auc:.2f})')
                    plt.plot([0, 1], [0, 1], color='navy',
                             lw=2, linestyle='--')
                    plt.xlim([0.0, 1.0])
                    plt.ylim([0.0, 1.05])
                    plt.xlabel('False Positive Rate')
                    plt.ylabel('True Positive Rate')
                    plt.title(f'Stage {self.stage} ROC Curve')
                    plt.legend(loc="lower right")
                    plt.savefig(
                        f"{self.log_folder}/viz/roc_stage{self.stage}.png")
                    plt.close()

                elif self.stage == 3:
                    # Per-class ROC curves (only if multilabel)
                    if y_true.ndim == 2 and y_prob.ndim == 2 and y_true.shape[1] == y_prob.shape[1]:
                        num_classes = y_true.shape[1]
                        fig, ax = plt.subplots(figsize=(10, 8))
                        for c in range(num_classes):
                            if len(np.unique(y_true[:, c])) == 2:
                                fpr, tpr, _ = roc_curve(
                                    y_true[:, c], y_prob[:, c])
                                roc_auc = auc(fpr, tpr)
                                title = OWASP_VULN[c] if c < len(
                                    OWASP_VULN) else f"Class {c}"
                                ax.plot(fpr, tpr, lw=2,
                                        label=f"{title} (AUC = {roc_auc:.2f})")
                        ax.plot([0, 1], [0, 1], color='navy',
                                lw=2, linestyle='--')
                        ax.set_xlim([0.0, 1.0])
                        ax.set_ylim([0.0, 1.05])
                        ax.set_xlabel('False Positive Rate')
                        ax.set_ylabel('True Positive Rate')
                        ax.set_title(
                            f'Stage {self.stage} ROC Curves per Class')
                        ax.legend(loc="lower right")
                        plt.savefig(
                            f"{self.log_folder}/viz/roc_per_class_stage{self.stage}.png")
                        plt.close()
                    else:
                        self.logger.warning(
                            "[visualize_results] Stage 3 ROC skipped (y_true/y_prob not multilabel 2D).")
        except Exception as e:
            self.logger.error(f"Visualization failed: {e}")
            traceback.print_exc()

    def train(self):
        """Wrapper to track time."""
        self.run_pre_train_analysis()
        super().train()
        self.training_time = time.time() - self.start_time

    def test(self):
        """Wrapper to visualize after test."""
        metrics = super().test()
        self.visualize_results(metrics)
        return metrics


class CascadeOrchestrator:
    """Manages the sequential training of the 3-Stage Pipeline."""

    def __init__(
        self,
        run_all_models=False,
        run_full=False,
        model_types=None,
        stage=None,
        batch_size=None,
        epochs=None,
        ablate_node_type=None,
        drop_cross_edges=False,
        keep_relation_types=None,
        drop_relation_types=None,
    ):
        self.results = {}
        self.stages = stage if stage is not None else [1, 2, 3]
        self.run_all_models = bool(run_all_models)
        self.run_full = bool(run_full)
        self.model_types = list(
            model_types) if model_types is not None else None
        self.batch_size = batch_size  # Use default batch sizes per stage in base_trainer
        self.epochs = epochs  # Use default epochs per stage in base_trainer
        self.ablate_node_type = ablate_node_type
        self.drop_cross_edges = bool(drop_cross_edges)
        self.keep_relation_types = keep_relation_types
        self.drop_relation_types = drop_relation_types

    def run(self):
        print("\n" + "#"*60)
        print("STARTING 3-STAGE CASCADE TRAINING PIPELINE")
        print("#"*60 + "\n")

        # Resolve which baseline variants to run.
        if self.run_full:
            try:
                from experiments.models.baseline_X import SUPPORTED_MODEL_TYPES
            except Exception:
                from models.baseline_X import SUPPORTED_MODEL_TYPES
            model_types = [None] + list(SUPPORTED_MODEL_TYPES)
        elif self.run_all_models:
            try:
                from experiments.models.baseline_X import SUPPORTED_MODEL_TYPES
            except Exception:
                from models.baseline_X import SUPPORTED_MODEL_TYPES
            model_types = list(SUPPORTED_MODEL_TYPES)
        else:
            model_types = self.model_types if self.model_types is not None else [
                None]

        for stage in self.stages:
            print(f"\n{'='*40}")
            print(f"LAUNCHING STAGE {stage}")
            print(f"{'='*40}")

            try:
                for model_type in model_types:
                    name = model_type if model_type is not None else "DEFAULT"
                    print(f"\n--- MODEL: {name} ---")

                    # 1. Initialize Trainer for this stage
                    # Stage 3 graphs are smaller but more nodes? Adjust as needed.
                    trainer = TrainerWithAnalysis(
                        stage=stage,
                        batch_size=self.batch_size,
                        num_epochs=self.epochs,
                        do_test_data=False,
                        model_type=model_type,
                        ablate_node_type=self.ablate_node_type,
                        drop_cross_edges=self.drop_cross_edges,
                        keep_relation_types=self.keep_relation_types,
                        drop_relation_types=self.drop_relation_types,
                    )

                    # 2. Train
                    trainer.train()

                    # 3. Test
                    metrics = trainer.test()

                    # 4. Store Results (keyed by stage + model)
                    self.results[(stage, name)] = {
                        "metrics": metrics,
                        "time": trainer.training_time,
                        "params": sum(p.numel() for p in trainer.model.parameters()),
                    }

                    # Free memory
                    del trainer
                    torch.cuda.empty_cache()
                    import gc
                    gc.collect()

            except Exception:
                print(f"!!! CRITICAL FAILURE IN STAGE {stage} !!!")
                traceback.print_exc()
                break

        self.print_summary()

    def print_summary(self):
        """Prints a comparison table of the cascade efficiency."""
        print("\n" + "="*80)
        print("CASCADE EFFICIENCY SUMMARY")
        print("="*80)

        # Header
        print(f"{'Stage':<10} | {'Role':<15} | {'F1':<8} | {'AUC':<8} | {'Hamming':<8} | {'Time (m)':<10} | {'Params':<10}")
        print("-" * 85)

        roles = {1: "Gatekeeper", 2: "Locator", 3: "Specialist"}

        for stage in self.stages:
            # Print in a stable order: stage, then model name.
            stage_rows = [(k, v)
                          for k, v in self.results.items() if k[0] == stage]
            stage_rows.sort(key=lambda kv: kv[0][1])
            if not stage_rows:
                continue

            for (s, model_name), data in stage_rows:
                m = data["metrics"]
                params = data["params"]

                time_mins = data["time"] / 60
                role = f"{roles[stage]}:{model_name}"
                print(
                    f"{stage:<10} | {role:<15} | {m['f1']:.4f}  | {m['auc']:.4f}  | {m.get('hamming', '--'):<8} | {time_mins:.1f}      | {params:,}"
                )

        print("-" * 75)
        print("="*80 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="ChainGuard training + analysis runner")
    parser.add_argument(
        "--run-all-models",
        action="store_true",
        help="Run every baseline variant implemented in experiments/models/baseline_X.py",
    )
    parser.add_argument(
        "--run-full",
        action="store_true",
        help="Run default model + all baseline variants.",
    )
    parser.add_argument(
        "--model-types",
        nargs="+",
        default=None,
        help="Optional explicit list of model types to run (overrides default when provided).",
    )
    parser.add_argument(
        "--stage",
        type=int,
        nargs='+',
        choices=[1, 2, 3],
        default=None,
        help="Specify stages to run (1, 2, or 3). Default runs all stages in cascade.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Optional batch size to use for all stages (overrides default when provided).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Optional number of epochs to use for all stages (overrides default when provided).",
    )
    parser.add_argument(
        "--ablate-node-type",
        type=str,
        default="none",
        choices=["none", "cfg", "ast"],
        help="Ablation: set one node type to 0 nodes (and drop incident edges) while keeping schema.",
    )
    parser.add_argument(
        "--drop-cross-edges",
        action="store_true",
        help="Ablation: drop only AST<->CFG cross edges (e.g., ast_to_cfg).",
    )
    parser.add_argument(
        "--keep-relations",
        type=str,
        default="",
        help="Relation-type pruning: comma-separated etype names to KEEP (e.g., cf,df,call).",
    )
    parser.add_argument(
        "--drop-relations",
        type=str,
        default="",
        help="Relation-type pruning: comma-separated etype names to DROP.",
    )
    args = parser.parse_args()

    orchestrator = CascadeOrchestrator(
        run_all_models=args.run_all_models,
        run_full=args.run_full,
        model_types=args.model_types,
        stage=args.stage,
        batch_size=args.batch_size,
        epochs=args.epochs,
        ablate_node_type=(None if args.ablate_node_type ==
                          "none" else args.ablate_node_type),
        drop_cross_edges=args.drop_cross_edges,
        keep_relation_types=(args.keep_relations or None),
        drop_relation_types=(args.drop_relations or None),
    )
    orchestrator.run()


# python experiments/main_trainer_analysis.py --stage 3 --ablate-node-type cfg ; python experiments/main_trainer_analysis.py --stage 3 --ablate-node-type ast ; python experiments/main_trainer_analysis.py --stage 3 --drop-cross-edges ; python experiments/main_trainer_analysis.py --stage 3 --drop-relations "df" ; python experiments/main_trainer_analysis.py --stage 3 --drop-relations "call,return_call" ; python experiments/main_trainer_analysis.py --stage 3 --ablate-node-type cfg --drop-relations "call,return_call"
