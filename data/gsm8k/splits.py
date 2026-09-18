"""One explicit split policy for training and experience construction."""
from __future__ import annotations


def split_gsm8k(raw_dataset, *, val_ratio: float = 0.1, seed: int = 42):
    """Preserve the official test set; split official train by row with a fixed seed."""
    if not 0 < val_ratio < 1:
        raise ValueError("val_ratio must be between zero and one")
    size = int(len(raw_dataset["train"]) * val_ratio)
    if not 0 < size < len(raw_dataset["train"]):
        raise ValueError("The validation split must contain at least one row")
    split = raw_dataset["train"].train_test_split(test_size=size, shuffle=True, seed=seed)
    return {"train": split["train"], "valid": split["test"], "test": raw_dataset["test"]}
