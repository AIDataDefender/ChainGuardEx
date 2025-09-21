"""
Training utilities for ChainGuard.
Handles model training, optimization, and checkpoint saving.
"""

import os
import logging
import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler
from transformers import AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm

logger = logging.getLogger(__name__)


def train(args, train_dataset, model, tokenizer, evaluate_func):
    """
    Train the model.

    Args:
        args: Training arguments
        train_dataset: Training dataset
        model: Model to train
        tokenizer: Tokenizer
        evaluate_func: Evaluation function
    """
    # Build dataloader
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(
        train_dataset,
        sampler=train_sampler,
        batch_size=args.train_batch_size,
        num_workers=4,
    )

    if len(train_dataset) == 0:
        print(
            "Warning: number of examples is not a multiple of the batch size. Last batch may be truncated."
        )
        return

    args.max_steps = args.epochs * len(train_dataloader)
    args.save_steps = len(train_dataloader) // 5
    print("save step", args.save_steps)
    args.warmup_steps = args.max_steps // 5
    model.to(args.device)

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(
        optimizer_grouped_parameters, lr=args.learning_rate * 0.1, eps=args.adam_epsilon
    )
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=args.max_steps
    )

    # Multi-gpu training
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)
        print(f"Using {args.n_gpu} gpu")

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.epochs)
    logger.info(
        "  Instantaneous batch size per GPU = %d",
        args.train_batch_size // max(args.n_gpu, 1),
    )
    logger.info(
        "  Total train batch size = %d",
        args.train_batch_size * args.gradient_accumulation_steps,
    )
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", args.max_steps)

    global_step = 0
    tr_loss, logging_loss, avg_loss, tr_nb, tr_num, train_loss = 0.0, 0.0, 0.0, 0, 0, 0
    best_f1 = 0

    model.zero_grad()

    for idx in range(args.epochs):
        bar = tqdm(train_dataloader, total=len(train_dataloader))
        tr_num = 0
        train_loss = 0
        for step, batch in enumerate(bar):
            (inputs_ids, position_idx, attn_mask, bytecode_embedding, labels) = [
                x.to(args.device) for x in batch
            ]
            model.train()
            loss, logits = model(
                inputs_ids, position_idx, attn_mask, bytecode_embedding, labels
            )

            if args.n_gpu > 1:
                loss = loss.mean()

            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            tr_loss += loss.item()
            tr_num += 1
            train_loss += loss.item()

            if avg_loss == 0:
                avg_loss = tr_loss

            avg_loss = round(train_loss / tr_num, 5)
            bar.set_description("epoch {} loss {}".format(idx, avg_loss))

            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
                global_step += 1
                avg_loss = round(
                    np.exp((tr_loss - logging_loss) / (global_step - tr_nb)), 4
                )

                if global_step % 500 == 0:
                    results = evaluate_func(
                        args, model, tokenizer, eval_when_training=True
                    )

                    # Save model checkpoint
                    if results["eval_f1"] > best_f1:
                        best_f1 = results["eval_f1"]
                        logger.info("  " + "*" * 20)
                        logger.info("  Best f1:%s", round(best_f1, 4))
                        logger.info("  " + "*" * 20)

                        checkpoint_prefix = "checkpoint-best-f1"
                        output_dir = os.path.join(
                            args.output_dir, "{}".format(checkpoint_prefix)
                        )
                        if not os.path.exists(output_dir):
                            os.makedirs(output_dir)
                        model_to_save = (
                            model.module if hasattr(model, "module") else model
                        )
                        output_dir = os.path.join(output_dir, "{}".format("model.bin"))
                        torch.save(model_to_save.state_dict(), output_dir)
                        logger.info("Saving model checkpoint to %s", output_dir)
                        logger.info("Saved model")
