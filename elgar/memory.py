"""One train-only predictive memory: observed prefix -> next item.

A single signed ridge map replaces basket reconstruction and the four-view mean.
No backbone scores, retrieval scores, validation labels, consumed-item mask,
user-specific gate, or hand-tuned internal fusion coefficient is used here.
"""
from __future__ import annotations

import json
import os
import tempfile
import time
from pathlib import Path

import numpy as np
from numba import njit
from scipy import linalg, sparse
from threadpoolctl import threadpool_limits
import torch

from .data import validate_histories
from .utils import training_digest, write_json
from .progress import progress, stage

RELATION_SCHEMA = "elgar-next-item-memory-v1"
DEFAULT_CACHE_ROOT = Path('cache')


def default_memory_cache(dataset_path: str | Path) -> Path:
    path = Path(dataset_path)
    name = path.parent.name if path.stem == 'dataset' else path.stem
    if not name:
        raise ValueError('Cannot determine dataset name for the memory cache.')
    return DEFAULT_CACHE_ROOT / f'{name}_memory'


def history_matrix(histories, num_items: int) -> sparse.csr_matrix:
    """Normalized reciprocal event-age features over the full observed prefix.

    Repeated IDs are summed by CSR construction. An event age is a position,
    not elapsed wall-clock time. No ID is masked at scoring time.
    """
    histories = validate_histories(histories, num_items)
    rows, cols, values = [], [], []
    for row, history in enumerate(histories):
        weights = 1.0 / np.arange(len(history), 0, -1, dtype=np.float32)
        weights /= weights.sum()
        rows.extend([row] * len(history))
        cols.extend((history - 1).tolist())
        values.extend(weights.tolist())
    return sparse.csr_matrix(
        (np.asarray(values, dtype=np.float32), (rows, cols)),
        shape=(len(histories), num_items), dtype=np.float32,
    )


@njit(cache=True, nogil=True)
def _fill_training(sequences, offsets, indptr, indices, values, targets):
    row = 0
    position = 0
    for user in range(len(offsets)-1):
        start, end = offsets[user], offsets[user+1]
        for length in range(1, end-start):
            indptr[row] = position
            denominator = np.float32(0)
            for age in range(1, length+1):
                denominator += np.float32(1.0/age)
            for event in range(length):
                indices[position] = sequences[start+event]-1
                values[position] = np.float32(1.0/(length-event))/denominator
                position += 1
            targets[row] = sequences[start+length]-1
            row += 1
    indptr[row] = position


def training_matrices(train: dict, num_items: int):
    """Every training event after the first contributes one causal (x, y) pair."""
    if not isinstance(train, dict) or not train:
        raise ValueError("Expected a nonempty training dictionary.")
    sequences = validate_histories([train[u] for u in sorted(train)], num_items)
    lengths = np.asarray([len(sequence) for sequence in sequences], np.int64)
    if np.any(lengths < 2):
        raise ValueError("Each source user must have at least two training events.")
    count = int(np.sum(lengths-1))
    nnz = int(np.sum(lengths*(lengths-1)//2))
    indptr = np.empty(count+1, np.int64)
    indices = np.empty(nnz, np.int32)
    values = np.empty(nnz, np.float32)
    targets = np.empty(count, np.int32)
    with stage('[memory] Building training samples'):
        _fill_training(np.concatenate(sequences), np.r_[0, np.cumsum(lengths)],
                       indptr, indices, values, targets)
    with stage('[memory] Building sparse matrices'):
        x = sparse.csr_matrix((values, indices, indptr), shape=(count, num_items), dtype=np.float32)
        x.sum_duplicates()
        x.sort_indices()
        y = sparse.csr_matrix(
            (np.ones(count, dtype=np.float32), (np.arange(count), targets)),
            shape=(count, num_items), dtype=np.float32,
        )
    return x, y


def ridge_map(x: sparse.csr_matrix, y: sparse.csr_matrix, device='cpu') -> np.ndarray:
    """Minimize ||XW-Y||_F^2 + sum_i (sum_t X_ti^2 + 1) ||W_i||_2^2.

    W is not constrained to be symmetric or to have a zero diagonal. Unlike
    auto-reconstruction, the next target is not an input event of its own pair.
    Diagonal and off-diagonal coefficients are both learned, never hard-masked.
    """
    if x.ndim != 2 or x.shape != y.shape or not x.shape[0]:
        raise ValueError("Training design and targets must be matching nonempty matrices.")
    with stage('[memory] Computing X.T @ X'):
        gram = (x.T @ x).toarray(order="F")
    with stage('[memory] Computing X.T @ Y'):
        rhs = (x.T @ y).toarray(order="F")
    n = gram.shape[0]
    diagonal = np.diag(gram).copy()
    gram.flat[::n + 1] += diagonal + 1.0
    if torch.device(device).type == 'cuda':
        a=torch.from_numpy(np.ascontiguousarray(gram)).to(device)
        torch.cuda.synchronize(device)
        with stage('[memory] GPU Cholesky factorization'):
            factor = torch.linalg.cholesky(a)
            torch.cuda.synchronize(device)
        # The factor replaces the Gram matrix for the solve. Release its GPU
        # input so large catalogues need three matrix buffers rather than four.
        del a
        with stage('[memory] Solving memory on GPU'):
            if rhs.nbytes >= 2**31:
                # Each RHS column is an independent solve with the same factor.
                # Two triangular solves avoid cuSOLVER's full factor copy.
                torch.cuda.empty_cache()
                weights = np.empty_like(rhs, order='F')
                for lo in range(0, n, 512):
                    hi = min(lo + 512, n)
                    b = torch.from_numpy(rhs[:, lo:hi]).to(device)
                    intermediate = torch.linalg.solve_triangular(factor, b, upper=False)
                    solution = torch.linalg.solve_triangular(factor.mT, intermediate, upper=True)
                    weights[:, lo:hi] = solution.cpu().numpy()
            else:
                b=torch.from_numpy(np.ascontiguousarray(rhs)).to(device)
                weights=torch.cholesky_solve(b,factor).cpu().numpy()
        if not np.isfinite(weights).all():
            raise FloatingPointError("Predictive memory contains nonfinite coefficients.")
        return weights
    # A fixed reduction environment keeps fresh fits reproducible in this build.
    with threadpool_limits(limits=2, user_api="blas"):
        with stage('[memory] CPU Cholesky factorization'):
            factor, info = linalg.lapack.spotrf(gram, lower=1, overwrite_a=1, clean=0)
        if info:
            raise np.linalg.LinAlgError(f"Predictive-memory Cholesky failed: info={info}")
        with stage('[memory] Solving memory on CPU'):
            weights, info = linalg.lapack.spotrs(factor, rhs, lower=1, overwrite_b=1)
        if info:
            raise np.linalg.LinAlgError(f"Predictive-memory solve failed: info={info}")
    if not np.isfinite(weights).all():
        raise FloatingPointError("Predictive memory contains nonfinite coefficients.")
    return weights


class ItemMemory:
    """One matrix, one query representation and one externally weighted output."""
    def __init__(self, matrix: np.ndarray, metadata: dict, device='cpu'):
        n = metadata.get("num_items")
        if type(n) is not int or n < 1 or matrix.shape != (n, n) or matrix.dtype != np.float32:
            raise ValueError("Invalid predictive-memory cache shape or dtype.")
        self.matrix, self.metadata, self.n = matrix, metadata, n
        self.device=torch.device(device)
        if self.device.type == 'cuda':
            self.matrix_gpu=torch.from_numpy(np.array(matrix,copy=True)).to(self.device)
            torch.cuda.synchronize(self.device)

    @classmethod
    def fit(cls, train: dict, num_items: int, folder: str | Path, device='cpu'):
        if type(num_items) is not int or num_items < 1:
            raise ValueError("num_items must be a positive integer.")
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        digest = training_digest(train, num_items)
        expected = {
            "schema": RELATION_SCHEMA,
            "training_digest": digest,
            "num_items": num_items,
            "objective": "all_causal_prefixes_to_one_next_item",
            "query": "full_prefix_reciprocal_event_age_l1_normalized",
            "regularizer": "diag(diag(X.T@X)+1)",
            "output": "signed_max_absolute_normalization",
            "diagonal": "learned_unconstrained_not_a_consumption_mask",
        }
        meta_path, matrix_path = folder / "metadata.json", folder / "predictive.npy"
        if meta_path.exists():
            saved = json.loads(meta_path.read_text(encoding="utf-8"))
            if any(saved.get(key) != value for key, value in expected.items()):
                raise ValueError("Memory version/training data mismatch; use a new cache directory.")
            print(f'[memory] Reusing cache: {folder}', flush=True)
            matrix = np.load(matrix_path, mmap_mode="r", allow_pickle=False)
            return cls(matrix, saved, device=device)
        started = time.perf_counter()
        print(f"[memory] next-item ridge: {num_items} items; one float32 matrix "
              f"{4*num_items*num_items/(1024**2):.1f} MiB (fit needs workspace)", flush=True)
        x, y = training_matrices(train, num_items)
        pairs = x.shape[0]
        weights = ridge_map(x, y, device=device)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(dir=folder, suffix=".tmp", delete=False) as handle:
                temporary = handle.name
            # CSR @ dense requires a C-contiguous RHS. A Fortran-layout cache
            # would otherwise copy the entire N x N matrix for EVERY query batch.
            mapped = np.lib.format.open_memmap(temporary, mode="w+", dtype=np.float32,
                                               shape=(num_items, num_items), fortran_order=False)
            for lo in progress(range(0, num_items, 128),desc='[memory] Saving cache',unit='batch'):
                mapped[lo:lo+128] = weights[lo:lo+128]
            with stage('[memory] Flushing cache to disk'):
                mapped.flush()
                mapped._mmap.close()
                with open(temporary, "rb") as handle:
                    os.fsync(handle.fileno())
                os.replace(temporary, matrix_path)
            temporary = None
        finally:
            if temporary is not None and os.path.exists(temporary):
                os.unlink(temporary)
        metadata = {**expected, "training_pairs": pairs, "seconds": time.perf_counter()-started}
        write_json(meta_path, metadata)
        return cls(np.load(matrix_path, mmap_mode="r", allow_pickle=False), metadata, device=device)

    def score(self, histories) -> np.ndarray:
        """All real items, no labels, no observed-item filtering, bounded correction."""
        if self.device.type == 'cuda':
            x=history_matrix(histories,self.n)
            # Sparse query vectors are small; the dense memory remains on GPU.
            query=torch.sparse_csr_tensor(torch.from_numpy(x.indptr.astype(np.int64)).to(self.device),
                torch.from_numpy(x.indices.astype(np.int64)).to(self.device),
                torch.from_numpy(x.data.copy()).to(self.device),size=x.shape,device=self.device)
            values=torch.sparse.mm(query,self.matrix_gpu)
            if not bool(torch.isfinite(values).all()):
                raise FloatingPointError("Nonfinite predictive-memory scores.")
            return values/values.abs().amax(dim=1,keepdim=True).clamp_min(1e-8)
        values = np.asarray(history_matrix(histories, self.n) @ self.matrix, dtype=np.float32)
        if not np.isfinite(values).all():
            raise FloatingPointError("Nonfinite predictive-memory scores.")
        denominator = np.maximum(np.max(np.abs(values), axis=1, keepdims=True), 1e-8)
        return values / denominator
