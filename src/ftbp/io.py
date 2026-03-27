# src/ftbp/io.py
from __future__ import annotations
from pathlib import Path
import pandas as pd
import numpy as np
import csv

def _read_csv_from_package(filename: str) -> pd.DataFrame:
    """
    Load a CSV that lives inside the ftbp package (next to this file),
    regardless of the caller's working directory.
    """
    # 1) Fast path: file is actually on disk next to this module
    here = Path(__file__).resolve()
    local = here.with_name(filename)
    if local.is_file():
        if filename == 'calcium.csv':
            vals: dict[str, np.ndarray] = {}
            with local.open("r", newline='') as f:
                for row in csv.reader(f):
                    if not row:
                        continue
                    label = row[0].strip().lower()
                    nums = [float(x) for x in row[1:] if x.strip() != ""]
                    if label in {"calcium", "placebo"}:
                        vals[label] = np.asarray(nums, dtype=float)

            if "calcium" not in vals or "placebo" not in vals:
                raise ValueError(f"Expected rows starting with 'calcium' and 'placebo' in {filename}.")
            return vals["calcium"], vals["placebo"]
        else:
            raise ValueError(f"Unknown file '{filename}' requested.")

def read_data(name: str) -> pd.DataFrame:
    """
    read_data('calcium') -> src/ftbp/calcium.csv
    Extend the mapping as needed.
    """
    mapping = {
        "calcium": "calcium.csv",
    }
    try:
        return _read_csv_from_package(mapping[name.lower()])
    except KeyError:
        raise ValueError(f"Unknown dataset '{name}'. Available: {sorted(mapping)}")
