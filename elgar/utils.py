"""Atomic artifacts, dataset fingerprints and ranking metrics."""
from __future__ import annotations
import hashlib
import json
import os
import random
import tempfile
from pathlib import Path
from typing import Any
import numpy as np
import torch
_THREAD_LIMITER = None


def write_json(path: str | Path, content: Any) -> None:
    """Write JSON atomically so interrupted runs never leave a valid-looking file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(content, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    temp_name = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent,
                                        prefix=f".{path.name}.", suffix=".tmp",
                                        delete=False) as handle:
            temp_name = handle.name
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
    finally:
        if temp_name is not None:
            try:
                os.unlink(temp_name)
            except FileNotFoundError:
                pass


def setup_runtime(seed: int = 2026, threads: int = 2, device: str = "auto") -> torch.device:
    if type(threads) is not int or type(seed) is not int or threads < 1 or seed < 0 or device not in ("auto", "cpu", "cuda"):
        raise ValueError("Invalid runtime configuration.")
    global _THREAD_LIMITER
    from threadpoolctl import threadpool_limits
    _THREAD_LIMITER = threadpool_limits(limits=threads)
    torch.set_num_threads(threads)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was explicitly requested but is not available.")
    chosen = "cuda" if device == "auto" and torch.cuda.is_available() else "cpu" if device == "auto" else device
    return torch.device(chosen)



def file_digest(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def training_digest(train, num_items: int) -> str:
    payload = {"num_items": num_items,
               "train": {str(u): list(map(int, train[u])) for u in sorted(train)}}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def vocabulary_digest(data) -> str:
    # Record original external IDs as well as internal positions. A matching
    # catalogue size alone cannot establish matching item semantics.
    mapping = sorted([(type(k).__name__, repr(k), int(v))
                      for k, v in data.raw["smap"].items()], key=lambda x: x[2])
    return hashlib.sha256(json.dumps(mapping, separators=(",", ":")).encode()).hexdigest()


def data_signature(data) -> dict:
    return {"num_items": data.num_items,
            "training_digest": training_digest(data.raw["train"], data.num_items),
            "vocabulary_digest": vocabulary_digest(data)}


def held_out_digest(data) -> str:
    payload = {split: [[user, data.raw[split][user]] for user in data.users]
               for split in ('val', 'test')}
    return hashlib.sha256(json.dumps(payload, separators=(',', ':')).encode()).hexdigest()


def ranking_metrics(ranks: np.ndarray | torch.Tensor) -> dict:
    r = ranks.detach().cpu().numpy() if isinstance(ranks, torch.Tensor) else np.asarray(ranks)
    if r.ndim != 1 or not len(r) or r.dtype.kind not in "iuf" or not np.isfinite(r).all() \
            or np.any(r < 1) or np.any(r != np.floor(r)):
        raise ValueError("Ranks must be a nonempty positive vector.")
    out = {"users": len(r), "hits@10": int((r <= 10).sum()), "MRR": float((1/r).mean())}
    for k in (1, 5, 10, 20):
        hit = r <= k
        out[f"HR@{k}"] = float(hit.mean())
        out[f"NDCG@{k}"] = float((hit/np.log2(r+1)).mean())
    return out


def stable_topk(scores: torch.Tensor, k: int = 20) -> torch.Tensor:
    if not isinstance(scores, torch.Tensor) or scores.ndim != 2 or type(k) is not int or k < 1:
        raise ValueError("scores must be a matrix and k must be a positive integer.")
    if not bool(torch.isfinite(scores).all()):
        raise ValueError("scores must be finite before ranking.")
    s = scores.clone()
    s[:, 0] = -torch.inf
    return torch.argsort(s, descending=True, dim=1, stable=True)[:, :min(k, s.shape[1]-1)]
