"""Dataset splitting utilities grouped by canonical SMILES.

Background: in the dataset, the same molecule appears repeatedly under different
device conditions (anchor_end / MetalOxide / control_PCE) (175 records, 154 unique
molecules). If row-level random splitting (train_test_split) is used, the same
molecule will fall into both the training and test sets, causing molecule-level
leakage and overly optimistic test metrics.

This module uses the RDKit canonical SMILES as the group id, ensuring that:
    all records of the same molecule appear on only one side of train or test.

Usage (in each subdirectory, e.g. delta_PCE/, conventionalML/):
    import sys; sys.path.insert(0, '..')
    from group_split import load_groups, group_train_test_split, group_cv, verify_disjoint

    x = data.iloc[:, :-1]
    y = data.iloc[:, -1]
    groups = load_groups('smiles.txt')                       # array of canonical SMILES
    train_idx, test_idx = group_train_test_split(x, y, groups, test_size=0.15, random_state=0)
    verify_disjoint(groups, train_idx, test_idx)             # assert no overlap
    cv = group_cv(groups, n_splits=10)                       # for use with GridSearchCV(cv=...)
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from sklearn.model_selection import GroupKFold, GroupShuffleSplit

RDLogger.DisableLog("rdApp.*")


def canonical_smiles(smiles) -> np.ndarray:
    """convert SMILES to canonical SMILES"""
    out = []
    for i, s in enumerate(smiles):
        mol = Chem.MolFromSmiles(s)
        if mol is None:
            raise ValueError(f"No. {i}  SMILES : Cannot parse {s!r}")
        out.append(Chem.MolToSmiles(mol))
    return np.asarray(out, dtype=object)


def load_groups(smiles_path: str = "smiles.txt") -> np.ndarray:
    """read molecules from smiles.txt and return canonical SMILES

    if the first row is title (e.g. 'SMILES'), it will be skipped
    """
    with open(smiles_path, "r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    if lines and lines[0].upper() == "SMILES":  
        lines = lines[1:]
    return canonical_smiles(lines)


def group_train_test_split(X, y, groups, test_size: float = 0.15, random_state: int = 0):
    """Split by molecule groups, returning (train_idx, test_idx)."""
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, test_idx = next(gss.split(X, y, groups))
    return train_idx, test_idx


def group_cv(groups, n_splits: int = 10):
    """Return a CV splitter grouped by molecule, ready to pass to GridSearchCV(cv=...) / cross_val_score."""
    return GroupKFold(n_splits=n_splits).split(np.zeros(len(groups)), groups=groups)


def verify_disjoint(groups, train_idx, test_idx) -> None:
    """Assert that no molecules are shared between train / test, and print group statistics."""
    tr = set(np.asarray(groups)[train_idx])
    te = set(np.asarray(groups)[test_idx])
    overlap = tr & te
    if overlap:
        raise AssertionError(f"train/test share {len(overlap)} molecules, split failed")
    uniq = len(set(np.asarray(groups)))
    print(f"molecues: tatal {uniq} | train {len(tr)} | test {len(te)} | overlap 0")
    print(f"samples: train {len(train_idx)} | test {len(test_idx)}")

def group_train_test_split(X, y, groups, test_size: float = 0.15, random_state: int = 0):
    
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, test_idx = next(gss.split(X, y, groups))
    return train_idx, test_idx

