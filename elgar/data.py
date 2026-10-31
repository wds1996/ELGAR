"""Validated leave-one-out data and observed evaluation histories.

Only plain-container pickle files are accepted. Item 0 is padding. The model
receives only an observed suffix. Full histories feed retrieval and memory.
"""
from __future__ import annotations

import pickle
import hashlib
from dataclasses import dataclass
from numbers import Integral
from pathlib import Path
from typing import Any, Mapping

import numpy as np


SPLITS = ("train", "val", "test")


def _plain_int(value: Any) -> bool:
    """Return true for integer IDs, excluding bool (a bool is an int in Python)."""
    return isinstance(value, Integral) and not isinstance(value, (bool, np.bool_))


class _PlainUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str) -> Any:
        raise pickle.UnpicklingError("Only plain Python containers are accepted.")


@dataclass
class DatasetBundle:
    raw: dict
    users: list[int]
    num_items: int
    format: str = "leave_one_out"
    source_sha256: str | None = None
    source_path: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.raw, dict) or not self.users or type(self.num_items) is not int:
            raise ValueError("Invalid dataset bundle metadata.")
        if self.num_items < 1 or self.format != "leave_one_out":
            raise ValueError("Invalid dataset bundle metadata.")


def load_dataset(path: str | Path) -> DatasetBundle:
    path = Path(path)
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"Dataset file does not exist: {path}")
    with path.open("rb") as handle:
        try:
            raw = _PlainUnpickler(handle).load()
        except (EOFError, pickle.UnpicklingError, ValueError, TypeError) as exc:
            raise ValueError(f"Could not read a plain-container dataset: {path}") from exc
    if not isinstance(raw, dict) or not {"train", "val", "test", "smap"} <= raw.keys():
        raise ValueError("Dataset must contain train, val, test and smap.")
    mapping = raw["smap"]
    if not isinstance(mapping, Mapping) or not mapping:
        raise ValueError("smap must be a nonempty dictionary.")
    n = len(mapping)
    if not all(_plain_int(v) for v in mapping.values()):
        raise ValueError("smap values must be integer item IDs.")
    # Normalize integral scalar subclasses (for example numpy.int64) once,
    # so downstream tensor construction never depends on pickle scalar types.
    mapping = {key: int(value) for key, value in mapping.items()}
    item_ids = set(mapping.values())
    zero_based = item_ids == set(range(n))
    if not zero_based and item_ids != set(range(1, n + 1)):
        raise ValueError("smap must map bijectively onto item IDs 0..N-1 or 1..N.")
    if not all(isinstance(raw[s], Mapping) for s in SPLITS):
        raise ValueError("Each split must be a user-to-sequence dictionary.")
    if not raw["train"] or set(raw["train"]) != set(raw["val"]) or set(raw["train"]) != set(raw["test"]):
        raise ValueError("All three splits must contain exactly the same users.")
    if not all(_plain_int(u) and 0 <= int(u) < 2**63 for u in raw["train"]):
        raise ValueError("User IDs must be nonnegative int64 integers.")
    users = sorted(int(u) for u in raw["train"])
    if len(set(users)) != len(users):
        raise ValueError("User IDs must be unique.")
    normalized_splits: dict[str, dict[int, list[int]]] = {}
    for split in SPLITS:
        normalized_splits[split] = {}
        for user in users:
            # Convert sequences to independent lists.  This prevents an
            # accidental in-place normalization from mutating nested pickle
            # objects shared by two splits.
            seq = raw[split][user]
            if not isinstance(seq, (list, tuple)) or not all(_plain_int(i) for i in seq):
                raise ValueError(f"Invalid item sequence: {split}, user={user}.")
            normalized_splits[split][user] = [int(i) for i in seq]
    for u in users:
        for split in SPLITS:
            seq = normalized_splits[split][u]
            if not all(i in item_ids for i in seq):
                raise ValueError(f"Invalid item sequence: {split}, user={u}.")
        if len(normalized_splits["train"][u]) < 2 or len(normalized_splits["val"][u]) != 1 or len(normalized_splits["test"][u]) != 1:
            raise ValueError(f"Invalid leave-one-out split sizes for user={u}.")
    if zero_based:
        # Shift every split and smap together: original item 0 is a real item,
        # while internal item 0 is reserved exclusively for padding.
        mapping = {key: value+1 for key, value in mapping.items()}
        normalized_splits = {
            split: {u: [i+1 for i in seq] for u, seq in values.items()}
            for split, values in normalized_splits.items()
        }
        print(f"[data] Converted zero-based item IDs 0..{n-1} to 1..{n}; "
              "0 is reserved for padding. Source file unchanged.", flush=True)
    normalized_raw = dict(raw)
    normalized_raw["smap"] = mapping
    normalized_raw.update(normalized_splits)
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            digest.update(block)
    return DatasetBundle(normalized_raw, users, n, source_sha256=digest.hexdigest(),
                         source_path=str(path.resolve()))


@dataclass
class EvaluationQueries:
    histories: list[np.ndarray]
    users: np.ndarray
    targets: np.ndarray
    split: str

    def __len__(self) -> int:
        return len(self.targets)


def make_evaluation_queries(data: DatasetBundle, split: str) -> EvaluationQueries:
    """Observed full histories and held-out targets without a model-specific tensor."""
    if not isinstance(data, DatasetBundle) or split not in ('val', 'test'):
        raise ValueError("Evaluation split must be val or test.")
    histories = []
    targets = np.empty(len(data.users), dtype=np.int64)
    for row, user in enumerate(data.users):
        train = np.asarray(data.raw['train'][user], dtype=np.int64)
        if split == 'test':
            history = np.concatenate((train, np.asarray(data.raw['val'][user], dtype=np.int64)))
        else:
            history = train
        history.setflags(write=False)
        histories.append(history)
        targets[row] = data.raw[split][user][0]
    return EvaluationQueries(histories, np.asarray(data.users, dtype=np.int64), targets, split)


def validate_histories(histories, num_items: int) -> list[np.ndarray]:
    """Check observed histories without accessing labels or dataset split state."""
    out = []
    for history in histories:
        h = np.asarray(history)
        if h.ndim != 1 or h.size == 0 or h.dtype.kind not in "iu":
            raise ValueError("Each observed history must be a nonempty integer vector.")
        if np.any(h < 1) or np.any(h > num_items):
            raise ValueError("Observed item IDs must be in 1..num_items.")
        out.append(h.astype(np.int64, copy=False))
    if not out:
        raise ValueError("An empty query batch is not supported.")
    return out
