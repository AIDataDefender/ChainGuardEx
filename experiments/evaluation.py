"""
Evaluation utilities for ChainGuard.
Handles model evaluation and testing with various metrics.
"""

import logging
from experiments.utils.logger import setup_logger
import numpy as np
import torch
from torch.utils.data import DataLoader, SequentialSampler
from sklearn.metrics import f1_score, recall_score, precision_score, accuracy_score
from tqdm import tqdm

from experiments.dataset import ModelDataset

logger = setup_logger("evaluation.log")


def evaluate(args, model, tokenizer, eval_when_training=False):
    """
    Evaluate the model.

    Args:
        args: Evaluation arguments
        model: Model to evaluate
        tokenizer: Tokenizer
        eval_when_training: Whether evaluation is during training

    Returns:
        dict: Evaluation results
    """
    # Build dataloader
    eval_dataset = ModelDataset(tokenizer, args, file_path=args.eval_data_file)
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(
        eval_dataset,
        sampler=eval_sampler,
        batch_size=args.eval_batch_size,
        num_workers=4,
    )

    # Multi-gpu evaluate
    if args.n_gpu > 1 and eval_when_training is False:
        model = torch.nn.DataParallel(model)

    # Eval!
    logger.info("***** Running evaluation *****")
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)

    eval_loss = 0.0
    nb_eval_steps = 0
    model.eval()
    logits = []
    y_trues = []

    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        (inputs_ids, position_idx, attn_mask, bytecode_embedding, labels) = [
            x.to(args.device) for x in batch
        ]
        with torch.no_grad():
            lm_loss, logit = model(
                inputs_ids, position_idx, attn_mask, bytecode_embedding, labels
            )
            eval_loss += lm_loss.mean().item()
            logits.append(logit.cpu().numpy())
            y_trues.append(labels.cpu().numpy())
        nb_eval_steps += 1

    # Calculate scores
    logits = np.concatenate(logits, axis=0)
    y_trues = np.concatenate(y_trues, axis=0)
    best_threshold = 0.5

    y_preds = logits
    recall = recall_score(y_trues, y_preds, average="weighted")
    precision = precision_score(y_trues, y_preds, average="weighted")
    f1 = f1_score(y_trues, y_preds, average="weighted")
    accuracy = accuracy_score(y_trues, y_preds)

    result = {
        "eval_recall": float(recall),
        "eval_precision": float(precision),
        "eval_f1": float(f1),
        "eval_accuracy": float(accuracy),
        "eval_threshold": best_threshold,
    }

    logger.info("***** Eval results *****")
    for key in sorted(result.keys()):
        logger.info("  %s = %s", key, str(round(result[key], 4)))

    return result


def test(args, model, tokenizer, best_threshold=0):
    """
    Test the model with comprehensive metrics.

    Args:
        args: Test arguments
        model: Model to test
        tokenizer: Tokenizer
        best_threshold: Threshold for predictions

    Returns:
        dict: Test results
    """
    # Build dataloader
    eval_dataset = ModelDataset(tokenizer, args, file_path=args.test_data_file)
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(
        eval_dataset,
        sampler=eval_sampler,
        batch_size=args.eval_batch_size,
        num_workers=4,
    )

    # Multi-gpu evaluate
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Test!
    logger.info("***** Running Test *****")
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)

    eval_loss = 0.0
    nb_eval_steps = 0
    model.eval()
    logits = []
    y_trues = []

    for batch in tqdm(eval_dataloader, desc="Testing:"):
        (inputs_ids, position_idx, attn_mask, bytecode_embedding, labels) = [
            x.to(args.device) for x in batch
        ]
        with torch.no_grad():
            lm_loss, logit = model(
                inputs_ids, position_idx, attn_mask, bytecode_embedding, labels
            )
            eval_loss += lm_loss.mean().item()
            logits.append(logit.cpu().numpy())
            y_trues.append(labels.cpu().numpy())
        nb_eval_steps += 1

    # Output result
    logits = np.concatenate(logits, axis=0)
    y_trues = np.concatenate(y_trues, axis=0)
    y_preds = logits

    # Calculate comprehensive metrics
    weighted_recall = recall_score(y_trues, y_preds, average="weighted")
    weighted_precision = precision_score(y_trues, y_preds, average="weighted")
    weighted_f1 = f1_score(y_trues, y_preds, average="weighted")

    micro_recall = recall_score(y_trues, y_preds, average="micro")
    micro_precision = precision_score(y_trues, y_preds, average="micro")
    micro_f1 = f1_score(y_trues, y_preds, average="micro")

    macro_recall = recall_score(y_trues, y_preds, average="macro")
    macro_precision = precision_score(y_trues, y_preds, average="macro")
    macro_f1 = f1_score(y_trues, y_preds, average="macro")
    accuracy = accuracy_score(y_trues, y_preds)

    result = {
        "Test_micro_recall": float(micro_recall),
        "Test_micro_precision": float(micro_precision),
        "Test_micro_f1": float(micro_f1),
        "Test_macro_recall": float(macro_recall),
        "Test_macro_precision": float(macro_precision),
        "Test_macro_f1": float(macro_f1),
        "Test_weighted_recall": float(weighted_recall),
        "Test_weighted_precision": float(weighted_precision),
        "Test_weighted_f1": float(weighted_f1),
        "Test_accuracy": float(accuracy),
    }

    logger.info("***** Test Result *****")
    for key in sorted(result.keys()):
        logger.info("  %s = %s", key, str(round(result[key], 4)))

    return result
