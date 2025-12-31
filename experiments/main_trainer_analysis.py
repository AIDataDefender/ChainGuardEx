import os
import time
import traceback
import argparse
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, recall_score, roc_curve, auc
import numpy as np
from sklearn.metrics import f1_score, precision_score, roc_auc_score
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
                            f"  DEBUG - Label dicts for first 3 GRAPHS (each graph is one contracts):")
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
                y_true = test_metrics["y_true"]
                y_pred = test_metrics["y_pred"]

            if self.stage == 3:
                # Multilabel Confusion Matrix
                num_classes = y_true.shape[1]
                fig, axes = plt.subplots(2, 4, figsize=(20, 10))
                axes = axes.flatten()
                for c in range(num_classes):
                    cm = confusion_matrix(
                        y_true[:, c], y_pred[:, c], labels=[0, 1])
                    disp = ConfusionMatrixDisplay(
                        cm, display_labels=["Safe", "Vuln"])
                    disp.plot(cmap="Blues", ax=axes[c], colorbar=False)
                    axes[c].set_title(f'{OWASP_VULN[c]} Confusion Matrix')
                plt.tight_layout()
                plt.savefig(f"{self.log_folder}/viz/cm_stage{self.stage}.png")
                plt.close()
            else:
                # Binary Confusion Matrix
                if y_true.ndim > 1:
                    y_true_flat = (y_true.sum(axis=1) > 0).astype(int)
                    y_pred_flat = (y_pred.sum(axis=1) > 0).astype(int)
                else:
                    y_true_flat = y_true
                    y_pred_flat = y_pred
                cm = confusion_matrix(y_true_flat, y_pred_flat, labels=[0, 1])
                disp = ConfusionMatrixDisplay(
                    cm, display_labels=["Safe", "Vuln"])
                fig, ax = plt.subplots(figsize=(6, 6))
                disp.plot(cmap="Blues", ax=ax, colorbar=False)
                plt.title(f"Stage {self.stage} Confusion Matrix")
                plt.savefig(f"{self.log_folder}/viz/cm_stage{self.stage}.png")
                plt.close()
                y_pred = test_metrics["y_pred"]
                num_classes = y_true.shape[1]

                f1_per_class = []
                prec_per_class = []
                rec_per_class = []
                auc_per_class = []

                for c in range(num_classes):
                    f1 = f1_score(y_true[:, c], y_pred[:, c], zero_division=0)
                    prec = precision_score(
                        y_true[:, c], y_pred[:, c], zero_division=0)
                    rec = recall_score(
                        y_true[:, c], y_pred[:, c], zero_division=0)
                    f1_per_class.append(f1)
                    prec_per_class.append(prec)
                    rec_per_class.append(rec)

                    # AUC if available
                    if "y_prob" in test_metrics:
                        y_prob = test_metrics["y_prob"]
                        if len(np.unique(y_true[:, c])) == 2:
                            auc_val = roc_auc_score(y_true[:, c], y_prob[:, c])
                            auc_per_class.append(auc_val)
                        else:
                            auc_per_class.append(0.5)
                    else:
                        auc_per_class.append(0.5)

                # Plot
                x = np.arange(num_classes)
                width = 0.2
                fig, ax = plt.subplots(figsize=(12, 6))
                ax.bar(x - width, f1_per_class, width,
                       label='F1', color='blue')
                ax.bar(x, prec_per_class, width,
                       label='Precision', color='green')
                ax.bar(x + width, rec_per_class, width,
                       label='Recall', color='red')
                ax.set_xlabel('Vulnerability Class')
                ax.set_ylabel('Score')
                ax.set_title(f'Stage {self.stage} Per-Class Metrics')
                ax.set_xticks(x)
                ax.set_xticklabels([OWASP_VULN[c] for c in range(num_classes)])
                ax.legend()
                plt.savefig(
                    f"{self.log_folder}/viz/per_class_metrics_stage{self.stage}.png")
                plt.close()

            # Save confusion matrix statistics
            if test_metrics and "y_pred" in test_metrics and "y_true" in test_metrics:
                cm_stats = {"stage": self.stage}
                y_true_orig = test_metrics["y_true"]
                y_pred_orig = test_metrics["y_pred"]

                if self.stage == 3:
                    # Per class
                    num_classes = y_true_orig.shape[1]
                    cm_matrices = []
                    f1_per_class = []
                    prec_per_class = []
                    rec_per_class = []
                    auc_per_class = []

                    for c in range(num_classes):
                        cm = confusion_matrix(
                            y_true_orig[:, c], y_pred_orig[:, c], labels=[0, 1])
                        cm_matrices.append(cm.tolist())
                        tn, fp, fn, tp = cm.ravel()
                        prec = tp / (tp + fp) if tp + fp > 0 else 0
                        rec = tp / (tp + fn) if tp + fn > 0 else 0
                        f1 = 2 * tp / (2 * tp + fp + fn) if 2 * \
                            tp + fp + fn > 0 else 0
                        f1_per_class.append(f1)
                        prec_per_class.append(prec)
                        rec_per_class.append(rec)

                        if "y_prob" in test_metrics:
                            y_prob = test_metrics["y_prob"]
                            if len(np.unique(y_true_orig[:, c])) == 2:
                                auc_val = roc_auc_score(
                                    y_true_orig[:, c], y_prob[:, c])
                            else:
                                auc_val = 0.5
                        else:
                            auc_val = 0.5
                        auc_per_class.append(auc_val)

                    cm_stats["per_class"] = {
                        "f1": f1_per_class,
                        "precision": prec_per_class,
                        "recall": rec_per_class,
                        "auc": auc_per_class,
                        "cm_matrices": cm_matrices
                    }

                # Overall binary cm
                if self.stage == 3:
                    y_true_flat = (y_true_orig.sum(axis=1) > 0).astype(int)
                    y_pred_flat = (y_pred_orig.sum(axis=1) > 0).astype(int)
                else:
                    y_true_flat = y_true_orig
                    y_pred_flat = y_pred_orig

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
                y_prob = test_metrics["y_prob"]

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
                    # Per-class ROC curves
                    num_classes = y_true.shape[1]
                    fig, ax = plt.subplots(figsize=(10, 8))
                    for c in range(num_classes):
                        if len(np.unique(y_true[:, c])) == 2:
                            fpr, tpr, _ = roc_curve(y_true[:, c], y_prob[:, c])
                            roc_auc = auc(fpr, tpr)
                            ax.plot(
                                fpr, tpr, lw=2, label=f'{OWASP_VULN[c]} (AUC = {roc_auc:.2f})')
                    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
                    ax.set_xlim([0.0, 1.0])
                    ax.set_ylim([0.0, 1.05])
                    ax.set_xlabel('False Positive Rate')
                    ax.set_ylabel('True Positive Rate')
                    ax.set_title(f'Stage {self.stage} ROC Curves per Class')
                    ax.legend(loc="lower right")
                    plt.savefig(
                        f"{self.log_folder}/viz/roc_per_class_stage{self.stage}.png")
                    plt.close()
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

    def __init__(self, run_all_models=False, model_types=None, stage=None, batch_size=None, epochs=None):
        self.results = {}
        self.stages = stage if stage is not None else [1, 2, 3]
        self.run_all_models = bool(run_all_models)
        self.model_types = list(
            model_types) if model_types is not None else None
        self.batch_size = batch_size  # Use default batch sizes per stage in base_trainer
        self.epochs = epochs  # Use default epochs per stage in base_trainer

    def run(self):
        print("\n" + "#"*60)
        print("STARTING 3-STAGE CASCADE TRAINING PIPELINE")
        print("#"*60 + "\n")

        # Resolve which baseline variants to run.
        if self.run_all_models:
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
    args = parser.parse_args()

    orchestrator = CascadeOrchestrator(
        run_all_models=args.run_all_models,
        model_types=args.model_types,
        stage=args.stage,
        batch_size=args.batch_size,
        epochs=args.epochs,
    )
    orchestrator.run()
