import logging
import os

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from transformers import EvalPrediction


def create_directory(dir_name: str):
    """lay out the directory structure"""
    if not os.path.exists(dir_name):
        logging.info(f"creating {dir_name} directory")
        os.makedirs(dir_name)
    return os.path.abspath(dir_name)


def read_csv(path: str, col_names=None) -> pd.DataFrame:
    """
    Read a csv file into a DataFrame.
    """
    return pd.read_csv(path, names=col_names) if col_names else pd.read_csv(path)

def compute_metrics(p: EvalPrediction):
    """
    Compute evaluation metrics for the model's predictions.
    """
    # Get the predicted class by taking the argmax of the logits
    preds = np.argmax(p.predictions, axis=1)
    # Calculate metrics by comparing the predicted classes to the true labels
    precision, recall, f1, _ = precision_recall_fscore_support(p.label_ids, preds, average='weighted')
    accuracy = accuracy_score(p.label_ids, preds)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }
