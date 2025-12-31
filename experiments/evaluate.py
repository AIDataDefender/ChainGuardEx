import os
import torch
import random
import numpy as np
import json
import matplotlib.pyplot as plt
import argparse
from torch.utils.data import Subset
from dgl.dataloading import GraphDataLoader

try:
    from experiments.dataset import CustomDataset, custom_collate
    from experiments.main_trainer_analysis import TrainerWithAnalysis
except ImportError:
    from dataset import CustomDataset, custom_collate
    from main_trainer_analysis import TrainerWithAnalysis

# Define model paths - UPDATE THESE PATHS TO YOUR LOG FOLDERS CONTAINING best_model.pth
model_paths = {
    1: "Logs/Stage1_20251230_1332_Our/best_model.pth",  # Update with actual log folder
    # 2: "Logs/Stage2_20251228_0414/best_model.pth",  # Update with actual log folder
    # 3: "Logs/Stage3_20251229_0958_Our/best_model.pth",  # Update with actual log folder
}

# Stages to evaluate


def main():
    for stage in stages:
        print(f"\n{'='*50}")
        print(f"Evaluating Stage {stage}")
        print(f"{'='*50}")

        # Create trainer to get model architecture and data split
        trainer = TrainerWithAnalysis(
            stage=stage,
            batch_size=2048 if stage == 3 else 768,
            num_epochs=1,
            log_folder=f"Eval_Stage{stage}",
            do_test_data=False  # Skip data analysis
        )

        # Load pre-trained model
        model_path = model_paths[stage]
        if os.path.exists(model_path):
            trainer.model.load_state_dict(torch.load(
                model_path, map_location=trainer.device))
            trainer.model.to(trainer.device)
            trainer.model.eval()
            print(f"Loaded model from {model_path}")
        else:
            print(f"Model path {model_path} not found. Skipping.")
            continue

        # Load history and redraw plot with adjusted y-limits
        log_folder = os.path.dirname(model_path)
        history_path = os.path.join(log_folder, "history.json")
        if os.path.exists(history_path):
            with open(history_path, 'r') as f:
                history = json.load(f)
            redraw_history_plot(history, stage, log_folder)
        else:
            print(f"History file {history_path} not found.")

        # Evaluate on the test set
        with torch.no_grad():
            metrics = trainer.evaluate(
                trainer.test_loader, fixed_thresholds=trainer.fixed_thresholds, optimize_thresholds=False)

        print(f"Test Metrics for Stage {stage}:")
        for k, v in metrics.items():
            if k not in ['y_true', 'y_pred', 'y_prob']:
                print(f"  {k}: {v}")

        # Visualize results (confusion matrix, etc.)
        trainer.visualize_results(metrics)

        # Log to file (exclude large arrays)
        main_metrics = {k: v for k, v in metrics.items() if k not in [
            'y_true', 'y_pred', 'y_prob']}
        trainer.logger.info(f"Evaluation: {main_metrics}")


def redraw_history_plot(history, stage, log_folder):
    """Redraw training history plot with adjusted y-limits."""
    try:
        os.makedirs(f"{log_folder}/viz", exist_ok=True)

        if history.get("train_loss"):
            fig, axes = plt.subplots(1, 2, figsize=(14, 6))

            # Loss
            axes[0].plot(history["train_loss"], label="Train")
            axes[0].plot(history["val_loss"], label="Val")
            axes[0].set_title(f"Stage {stage} Loss")
            axes[0].set_xlabel("Epoch")
            axes[0].set_ylabel("Loss")
            axes[0].set_ylim(0, 0.5)  # Fixed y-axis for loss
            axes[0].legend()

            # Metrics
            axes[1].plot(history["val_f1"],
                         label="Macro F1", color="green")
            axes[1].plot(history["val_auc"],
                         label="AUC", color="purple")
            if stage == 3 and "val_hamming" in history:
                axes[1].plot(history["val_hamming"],
                             label="Hamming Score", color="orange")
            axes[1].set_title("Validation Metrics")
            axes[1].set_xlabel("Epoch")
            # Adjust y-axis for metrics: min(0.5, lowest) to 1
            all_metrics = history["val_f1"] + history["val_auc"]
            if stage == 3 and "val_hamming" in history:
                all_metrics += history["val_hamming"]
            min_val = min(all_metrics) if all_metrics else 0
            axes[1].set_ylim(min(0.5, min_val), 1)
            axes[1].legend()

            plt.savefig(
                f"{log_folder}/viz/history_stage{stage}_redrawn.png")
            plt.close()

    except Exception as e:
        print(f"History plot redraw failed: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate ChainGuard models")
    parser.add_argument(
        "--stage",
        type=int,
        nargs='+',
        choices=[1, 2, 3],
        default=[1, 2, 3],
        help="Specify stages to evaluate (1, 2, or 3). Default evaluates all stages.",
    )
    parser.add_argument(
        "--model-path-1",
        type=str,
        default=None,
        help="Path to stage 1 model. If not provided, uses default.",
    )
    parser.add_argument(
        "--model-path-2",
        type=str,
        default=None,
        help="Path to stage 2 model. If not provided, uses default.",
    )
    parser.add_argument(
        "--model-path-3",
        type=str,
        default=None,
        help="Path to stage 3 model. If not provided, uses default.",
    )
    args = parser.parse_args()
    # Update model paths if provided
    if args.model_path_1:
        model_paths[1] = args.model_path_1
    if args.model_path_2:
        model_paths[2] = args.model_path_2
    if args.model_path_3:
        model_paths[3] = args.model_path_3
    stages = args.stage
    main()
