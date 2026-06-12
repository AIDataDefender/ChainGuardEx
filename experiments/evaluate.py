import os
import torch
import json
import matplotlib.pyplot as plt
import argparse


def _infer_hidden_dim_from_state_dict(state_dict, stage):
    """Best-effort hidden_dim inference from saved weights.

    Your checkpoints save only model.state_dict(), so evaluate() must
    reconstruct the same architecture/dimensions to load it.
    """
    if not isinstance(state_dict, dict):
        return None
    w = state_dict.get("classifier.0.weight")
    if isinstance(w, torch.Tensor) and w.ndim == 2:
        if stage == 3:
            # Stage 3: classifier.0 is Linear(hidden_dim, hidden_dim//2), shape (hidden_dim//2, hidden_dim)
            return int(w.shape[1])
        else:
            # Stage 1/2: classifier.0 is Linear(hidden_dim*2, hidden_dim), shape (hidden_dim, hidden_dim*2)
            return int(w.shape[0])
    return None


def _extract_rel_keys_from_state_dict(state_dict):
    """Extract relation keys from a model state_dict.

    Supports both RGCN and HGINA stacks by scanning keys like:
      - rgcn.layers.X.weight.<rel_key>.weight
      - hgina_stack.layers.X.<rel_key>.lin_src.weight
    """
    if not isinstance(state_dict, dict):
        return []

    rel_keys = set()
    for k in state_dict.keys():
        # RGCN case
        if k.startswith("rgcn.layers.") and ".weight." in k and k.endswith(".weight"):
            # rgcn.layers.0.weight.<rel_key>.weight
            parts = k.split(".weight.")
            if len(parts) == 2 and parts[1].endswith(".weight"):
                rel_key = parts[1][: -len(".weight")]
                if rel_key:
                    rel_keys.add(rel_key)
            continue

        # HGINA case
        if k.startswith("hgina_stack.layers.") and ".lin_src.weight" in k:
            # hgina_stack.layers.0.<rel_key>.lin_src.weight
            prefix = "hgina_stack.layers."
            tail = k[len(prefix):]
            parts = tail.split(".")
            if len(parts) >= 3:
                rel_key = parts[1]
                if rel_key:
                    rel_keys.add(rel_key)

    return sorted(rel_keys)


def _rel_keys_to_rel_names(rel_keys, ntypes):
    """Convert rel_key strings into canonical rel tuples using known node types."""
    if not rel_keys or not ntypes:
        return []

    ntypes = sorted(set(ntypes), key=lambda s: -len(s))
    rel_names = []
    for rel_key in rel_keys:
        found = False
        for src in ntypes:
            src_prefix = f"{src}_"
            if not rel_key.startswith(src_prefix):
                continue
            for dst in ntypes:
                dst_suffix = f"_{dst}"
                if not rel_key.endswith(dst_suffix):
                    continue
                etype = rel_key[len(src_prefix): -len(dst_suffix)]
                if etype:
                    rel_names.append((src, etype, dst))
                    found = True
                    break
            if found:
                break
    return rel_names


def _infer_model_type_from_tag(model_tag: str):
    """Infer BaseTrainer.model_type from the run folder name.

    model_type drives BaseTrainer.setup_model():
    - None -> experiments.models.proto_1 (HGINA+RGCN)
    - non-None -> experiments.models.baseline_X with that model_type
    """
    if not model_tag:
        return None
    t = model_tag.upper()

    # Keep these checks conservative to avoid accidentally selecting baseline_X
    # for custom/"Our" runs.
    if "HGINA" in t or "HGIN" in t or "RGCN" in t or "PROTO" in t or "OUR" in t:
        return None

    if "GATV2" in t:
        return "GATv2_no_edge"
    if "HGT" in t:
        return "HGT"
    if "SAGE" in t:
        return "GraphSAGE"
    if "GCN" in t:
        return "GCN"
    # Must come after HGIN/HGINA guard
    if "GIN" in t:
        return "GIN"

    return None


try:
    from experiments.main_trainer_analysis import TrainerWithAnalysis
except ImportError:
    from main_trainer_analysis import TrainerWithAnalysis

# Define model paths (per stage) - each stage maps to a list of model checkpoints.
# You can evaluate multiple models for the same stage by adding more paths.
model_paths = {
    1: [
#         "Logs/Stage1_20260114_1845_ASTabla/best_model.pth",
# "Logs/Stage1_20260114_2213_CFGabla/best_model.pth"
"Logs/Stage1_20260117_0127/best_model.pth"
    ],
    2: [
        "Logs/Stage2_20260101_2045_GIN/best_model.pth",
        "Logs/Stage2_20260101_1422_SAGE/best_model.pth",
        "Logs/Stage2_20251231_1947_GCN/best_model.pth",
    ],
    3: [
# "Logs/Stage3_20260114_1805_ASTabla/best_model.pth",
# "Logs/Stage3_20260114_1751_CFGabla/best_model.pth"
"Logs/Stage3_20260117_0121/best_model.pth"
    ],
}

# Stages to evaluate


def main():
    for stage in stages:
        print(f"\n{'='*50}")
        print(f"Evaluating Stage {stage}")
        print(f"{'='*50}")

        stage_model_paths = model_paths.get(stage, [])
        if not stage_model_paths:
            print(f"No model paths configured for Stage {stage}. Skipping.")
            continue

        # Build dataset/splits/loaders ONCE per stage.
        # We'll swap model architectures/weights per checkpoint without reloading data.
        stage_trainer = TrainerWithAnalysis(
            stage=stage,
            batch_size=2048 if stage == 3 else 768,
            num_epochs=1,
            log_folder=f"Eval_Stage{stage}",
            do_test_data=False,  # Skip data analysis
        )

        for model_path in stage_model_paths:
            model_path = os.path.normpath(model_path)
            model_tag = os.path.basename(
                os.path.dirname(model_path)) or f"Stage{stage}"

            print(f"\n--- Model: {model_tag} ---")

            # Load checkpoint weights first so we can infer hidden_dim and choose the right model.
            if not os.path.exists(model_path):
                print(f"Model path {model_path} not found. Skipping.")
                continue

            state_dict = torch.load(model_path, map_location="cpu")
            inferred_hidden_dim = _infer_hidden_dim_from_state_dict(state_dict, stage)
            inferred_model_type = _infer_model_type_from_tag(model_tag)
            if inferred_model_type is not None:
                print(f"[EVAL] Inferred model_type={inferred_model_type}")
            else:
                print("[EVAL] Inferred model_type=None (proto_1)")
            if inferred_hidden_dim is not None:
                print(f"[EVAL] Inferred hidden_dim={inferred_hidden_dim}")

            # Align rel_names to checkpoint (handles ablated training runs)
            try:
                rel_keys = _extract_rel_keys_from_state_dict(state_dict)
                if rel_keys:
                    ntypes = set()
                    for r in (stage_trainer.dataset1.rel_names or []):
                        try:
                            ntypes.add(r[0])
                            ntypes.add(r[2])
                        except Exception:
                            pass
                    inferred_rels = _rel_keys_to_rel_names(rel_keys, ntypes)
                    if inferred_rels:
                        stage_trainer.dataset1.rel_names = inferred_rels
                        print(f"[EVAL] Using rel_names from checkpoint ({len(inferred_rels)} rels)")
            except Exception as e:
                print(f"[EVAL] rel_names inference failed: {e}")

            # Rebuild the model for this checkpoint (without reloading data)
            stage_trainer.model_type = inferred_model_type
            if inferred_hidden_dim is not None:
                stage_trainer.hidden_dim = inferred_hidden_dim
            stage_trainer.setup_model()

            # Load pre-trained weights
            stage_trainer.model.load_state_dict(state_dict)
            stage_trainer.model.to(stage_trainer.device)
            stage_trainer.model.eval()
            print(f"Loaded model from {model_path}")

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
                metrics = stage_trainer.evaluate(
                    stage_trainer.test_loader,
                    fixed_thresholds=stage_trainer.fixed_thresholds,
                    optimize_thresholds=False,
                )

            print(f"Test Metrics for Stage {stage} ({model_tag}):")
            for k, v in metrics.items():
                if k not in ['y_true', 'y_pred', 'y_prob']:
                    print(f"  {k}: {v}")

            # Visualize results (confusion matrix, etc.)
            # Note: visuals are saved under the stage-level eval folder.
            stage_trainer.visualize_results(metrics)

            # Log to file (exclude large arrays)
            main_metrics = {k: v for k, v in metrics.items() if k not in [
                'y_true', 'y_pred', 'y_prob']}
            stage_trainer.logger.info(
                f"[{model_tag}] Evaluation: {main_metrics}")


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
        nargs='*',
        default=None,
        help="Path to stage 1 model. If not provided, uses default.",
    )
    parser.add_argument(
        "--model-path-2",
        type=str,
        nargs='*',
        default=None,
        help="Path to stage 2 model. If not provided, uses default.",
    )
    parser.add_argument(
        "--model-path-3",
        type=str,
        nargs='*',
        default=None,
        help="Path to stage 3 model. If not provided, uses default.",
    )
    args = parser.parse_args()
    # Update model paths if provided
    if args.model_path_1 is not None and len(args.model_path_1) > 0:
        model_paths[1] = args.model_path_1
    if args.model_path_2 is not None and len(args.model_path_2) > 0:
        model_paths[2] = args.model_path_2
    if args.model_path_3 is not None and len(args.model_path_3) > 0:
        model_paths[3] = args.model_path_3
    stages = args.stage
    main()
