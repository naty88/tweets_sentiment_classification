import logging
import os
from typing import List

import pandas as pd


def create_directory(dir_name: str):
    """lay out the directory structure"""
    if not os.path.exists(dir_name):
        logging.info(f"creating {dir_name} directory")
        os.makedirs(dir_name)
    return os.path.abspath(dir_name)


def read_csv(path: str, col_names: List[str]) -> pd.DataFrame:
    """
    Read a csv file into a DataFrame.
    """
    return pd.read_csv(path, names=col_names)
