"""Adapter for the original CoMER SASRec implementation and native checkpoints.

Uses the bundled model, independently of ReAd/RecBole.
"""
from __future__ import annotations
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from elgar.data import validate_histories
from elgar.utils import data_signature, held_out_digest


def _run_arguments(checkpoint: Path) -> dict:
    path = checkpoint.parent / 'args.txt'
    if not path.is_file():
        raise FileNotFoundError(f'Native SASRec args.txt is required beside the checkpoint: {path}')
    values = {}
    for line in path.read_text(encoding='utf-8').splitlines():
        key, value = line.split(',', 1)
        values[key] = value
    required = ('dataset', 'maxlen', 'hidden_units', 'num_blocks', 'num_heads',
                'dropout_rate', 'norm_first')
    if any(key not in values for key in required):
        raise ValueError('Incomplete native SASRec args.txt.')
    return values


class SASRecAdapter:
    """Load an unmodified native SASRec state_dict and score every item."""

    def _model_class(self):
        from .model import SASRec
        return SASRec

    def __init__(self, checkpoint, data, device='cpu'):
        checkpoint = Path(checkpoint)
        self.device = torch.device(device)
        self.num_items = data.num_items
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        native = _run_arguments(checkpoint)
        manifest_path = checkpoint.parent / 'data.manifest.json'
        if not manifest_path.is_file():
            raise FileNotFoundError(f'Dataset fingerprint is required beside the checkpoint: {manifest_path}')
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        if (not isinstance(manifest, dict) or type(manifest.get('num_users')) is not int
                or manifest['num_users'] != len(data.users)):
            raise ValueError('Native SASRec manifest must record the matching user count.')
        if manifest.get('schema') in ('sasrec-native-data-v2', 'sasrec-pickle-data-v1'):
            matching = (isinstance(manifest.get('source_sha256'), str)
                        and len(manifest['source_sha256']) == 64
                        and manifest.get('source_sha256') == data.source_sha256
                        and manifest.get('num_items') == data.num_items
                        and manifest.get('num_users') == len(data.users))
        elif manifest.get('schema') in ('elgar-sasrec-data-v1', 'v3-comer-sasrec-export-v1'):
            matching = (manifest.get('signature') == data_signature(data)
                        and manifest.get('held_out_digest') == held_out_digest(data))
        else:
            matching = False
        if not matching:
            raise ValueError('Native SASRec data does not match this dataset and item mapping.')
        self.max_len = int(native['maxlen'])
        if self.max_len < 1:
            raise ValueError('Invalid native SASRec sequence length.')
        args = SimpleNamespace(
            device=str(self.device), maxlen=self.max_len,
            hidden_units=int(native['hidden_units']), num_blocks=int(native['num_blocks']),
            num_heads=int(native['num_heads']), dropout_rate=float(native['dropout_rate']),
            norm_first=native['norm_first'] == 'True',
            padding_mask=native.get('padding_mask', 'False') == 'True',
            architecture=native.get('architecture', 'legacy'),
        )
        model = self._model_class()(manifest['num_users'], data.num_items, args)
        try:
            weights = torch.load(checkpoint, map_location='cpu', weights_only=True)
            model.load_state_dict(weights, strict=True)
        except (RuntimeError, ValueError, TypeError, KeyError) as exc:
            raise ValueError('Native SASRec checkpoint and args.txt are incompatible.') from exc
        if any(not bool(torch.isfinite(parameter).all()) for parameter in model.parameters()):
            raise ValueError('Native SASRec checkpoint contains nonfinite weights.')
        self.model = model.to(self.device).eval().requires_grad_(False)

    @torch.inference_mode()
    def score(self, histories) -> torch.Tensor:
        histories = validate_histories(histories, self.num_items)
        items = np.zeros((len(histories), self.max_len), dtype=np.int64)
        for row, history in enumerate(histories):
            suffix = history[-self.max_len:]
            items[row, -len(suffix):] = suffix
        state = self.model.log2feats(items)[:, -1, :]
        scores = state @ self.model.item_emb.weight[1:].T
        if hasattr(self.model, 'bias'):
            scores = scores + self.model.bias[1:]
        return scores


