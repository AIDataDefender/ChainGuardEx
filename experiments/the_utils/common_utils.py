"""
Common utility functions for ChainGuard.
Contains helper functions for dataset operations, seed setting, and other utilities.
"""

import pickle
import random
import numpy as np
import torch
import joblib


def saved_extract_dataset(dataset, filepath):
    """
    Save dataset to pickle file.

    Args:
        dataset: Dataset to save
        filepath: Path to save the dataset
    """
    with open(filepath, "wb") as f:
        pickle.dump(dataset, f)


def load_extract_dataset(filepath):
    """
    Load dataset from pickle file.

    Args:
        filepath: Path to load the dataset from

    Returns:
        Dataset: Loaded dataset
    """
    with open(filepath, "rb") as f:
        return pickle.load(f)


def load_dataset_from_pickle(file_path):
    """
    Load dataset using joblib.

    Args:
        file_path: Path to the pickle file

    Returns:
        Dataset: Loaded dataset
    """
    dataset = joblib.load(file_path)
    return dataset


def set_seed(args):
    """
    Set random seeds for reproducibility.

    Args:
        args: Arguments containing seed value
    """
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)
