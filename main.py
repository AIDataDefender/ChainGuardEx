"""
Main entry point for ChainGuard.
Handles argument parsing, model initialization, and orchestrates training/evaluation/testing.
"""

from __future__ import absolute_import, division, print_function

import argparse
import logging
import os
import torch
from transformers import (
    RobertaConfig,
    RobertaForSequenceClassification,
    RobertaTokenizer,
)

from experiments.model import Model
from experiments.dataset import ModelDataset
from experiments.training import train
from experiments.evaluation import evaluate, test
from experiments.utils.common_utils import set_seed

# Environment setup
TF_ENABLE_ONEDNN_OPTS = 0
logger = logging.getLogger(__name__)


###########################
#       CONFIG
# Defines default vars for model
###########################


###########################




def main():
    """Main function that orchestrates the entire pipeline."""
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "--train_data_file",
        default=None,
        type=str,
        help="The input training data file (a text file).",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        required=True,
        help="The output directory where the model predictions and checkpoints will be written.",
    )

    # Other parameters
    parser.add_argument(
        "--eval_data_file",
        default=None,
        type=str,
        help="An optional input evaluation data file to evaluate the perplexity on (a text file).",
    )
    parser.add_argument(
        "--test_data_file",
        default=None,
        type=str,
        help="An optional input evaluation data file to evaluate the perplexity on (a text file).",
    )

    parser.add_argument(
        "--model_name_or_path",
        default=None,
        type=str,
        help="The model checkpoint for weights initialization.",
    )

    parser.add_argument(
        "--config_name",
        default="",
        type=str,
        help="Optional pretrained config name or path if not the same as model_name_or_path",
    )
    parser.add_argument(
        "--tokenizer_name",
        default="",
        type=str,
        help="Optional pretrained tokenizer name or path if not the same as model_name_or_path",
    )

    parser.add_argument(
        "--code_length",
        default=256,
        type=int,
        help="Optional Code input sequence length after tokenization.",
    )
    parser.add_argument(
        "--data_flow_length",
        default=64,
        type=int,
        help="Optional Data Flow input sequence length after tokenization.",
    )
    parser.add_argument(
        "--do_train", action="store_true", help="Whether to run training."
    )
    parser.add_argument(
        "--do_eval", action="store_true", help="Whether to run eval on the dev set."
    )
    parser.add_argument(
        "--do_test", action="store_true", help="Whether to run eval on the dev set."
    )
    parser.add_argument(
        "--evaluate_during_training",
        action="store_true",
        help="Run evaluation during training at each logging step.",
    )
    parser.add_argument(
        "--train_batch_size",
        default=2,
        type=int,
        help="Batch size per GPU/CPU for training.",
    )
    parser.add_argument(
        "--eval_batch_size",
        default=4,
        type=int,
        help="Batch size per GPU/CPU for evaluation.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--learning_rate",
        default=5e-5,
        type=float,
        help="The initial learning rate for Adam.",
    )
    parser.add_argument(
        "--weight_decay", default=0.0, type=float, help="Weight deay if we apply some."
    )
    parser.add_argument(
        "--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer."
    )
    parser.add_argument(
        "--max_grad_norm", default=1.0, type=float, help="Max gradient norm."
    )
    parser.add_argument(
        "--max_steps",
        default=-1,
        type=int,
        help="If > 0: set total number of training steps to perform. Override num_train_epochs.",
    )
    parser.add_argument(
        "--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps."
    )

    parser.add_argument(
        "--seed", type=int, default=42, help="random seed for initialization"
    )
    parser.add_argument("--epochs", type=int, default=1, help="training epochs")

    args = parser.parse_args()

    # Setup CUDA, GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count() if torch.cuda.is_available() else 0
    args.save_steps = 500
    args.device = device

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.warning("device: %s, n_gpu: %s", device, args.n_gpu)

    # Set seed
    set_seed(args)

    # Initialize model configuration and tokenizer
    config = RobertaConfig.from_pretrained(
        args.config_name if args.config_name else args.model_name_or_path
    )
    config.num_labels = 10
    tokenizer = RobertaTokenizer.from_pretrained(
        args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
        clean_up_tokenization_spaces=False,
    )

    # Initialize model
    roberta_model = RobertaForSequenceClassification.from_pretrained(
        args.model_name_or_path, config=config
    )
    model = Model(roberta_model, config, tokenizer, args)

    logger.info("Training/evaluation parameters %s", args)

    # Training
    if args.do_train:
        train_dataset = ModelDataset(tokenizer, args, file_path=args.train_data_file)
        train(args, train_dataset, model, tokenizer, evaluate)

    # Evaluation
    results = {}
    if args.do_eval:
        checkpoint_prefix = "checkpoint-best-f1/model.bin"
        output_dir = os.path.join(args.output_dir, "{}".format(checkpoint_prefix))
        model.load_state_dict(torch.load(output_dir, weights_only=True))
        model.to(args.device)
        results = evaluate(args, model, tokenizer)

    if args.do_test:
        checkpoint_prefix = "checkpoint-best-f1/model.bin"
        output_dir = os.path.join(args.output_dir, "{}".format(checkpoint_prefix))
        model.load_state_dict(torch.load(output_dir, weights_only=True))
        model.to(args.device)
        test(args, model, tokenizer, best_threshold=0.5)

    return results


if __name__ == "__main__":
    main()
