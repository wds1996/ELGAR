"""Use an already loaded RecBole sequential model as an ELGAR backbone."""
from __future__ import annotations

import numpy as np
import torch

from elgar.data import validate_histories


class RecBoleBackboneAdapter:
    def __init__(self, model, num_items: int, max_len: int, device='cpu'):
        if type(num_items) is not int or num_items < 1 or type(max_len) is not int or max_len < 1:
            raise ValueError('num_items and max_len must be positive integers.')
        if model.n_items != num_items + 1:
            raise ValueError('RecBole and ELGAR item catalogues have different sizes.')
        self.model = model.to(device).eval().requires_grad_(False)
        self.num_items = num_items
        self.max_len = max_len
        self.device = torch.device(device)

    @torch.inference_mode()
    def score(self, histories) -> torch.Tensor:
        histories = validate_histories(histories, self.num_items)
        lengths_cpu = np.asarray([min(len(h), self.max_len) for h in histories], dtype=np.int64)
        items_cpu = np.zeros((len(histories), self.max_len), dtype=np.int64)
        for row, history in enumerate(histories):
            suffix = np.asarray(history[-self.max_len:], dtype=np.int64)
            items_cpu[row, :len(suffix)] = suffix
        lengths = torch.as_tensor(lengths_cpu, device=self.device)
        items = torch.as_tensor(items_cpu, device=self.device)
        state = self.model.forward(items, lengths)
        return state @ self.model.item_embedding.weight[1:].T
