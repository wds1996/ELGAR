"""Load ReAd's frozen DuoRec using already prepared ELGAR data.

No interaction-file reader, downloader, filtering or split generation belongs
in this module. Prepare data explicitly with ``python -m dataset.prepare``.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import sys

import torch

from elgar.utils import data_signature, file_digest, held_out_digest
from .recbole import RecBoleBackboneAdapter


@dataclass
class ReadBackbone:
    adapter: RecBoleBackboneAdapter
    seed: int
    checkpoint_dataset: str


class _Catalogue:
    """Minimal read-only metadata required by ReAd's DuoRec constructor."""
    def __init__(self, data, config):
        self.counts = {config['ITEM_ID_FIELD']: data.num_items + 1,
                       config['USER_ID_FIELD']: max(data.users) + 1}

    def num(self, field):
        return self.counts[field]


def _use_read(root):
    root = Path(root).resolve()
    package = root / 'recbole'
    if not (package / 'config/__init__.py').is_file():
        raise FileNotFoundError(f'ReAd RecBole implementation not found: {package}')
    loaded = sys.modules.get('recbole')
    if loaded is not None and Path(loaded.__file__).resolve().parent != package:
        raise RuntimeError('A different RecBole is already imported; start a fresh '
                           'Python process and supply the ReAd checkout.')
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))


def load_duorec(checkpoint, read_root, data, *, device='cpu', dataset_path):
    """Load only model/configuration and verify prepared-data provenance."""
    dataset_path = Path(dataset_path).resolve()
    manifest_path = dataset_path.with_name('preprocessing.json')
    if not manifest_path.is_file():
        raise FileNotFoundError(f'{manifest_path} is required. Run dataset.prepare '
                                'before loading DuoRec; runtime does not preprocess data.')
    manifest = json.loads(manifest_path.read_text())
    if (not isinstance(manifest, dict) or manifest.get('source') != 'recbole'
            or manifest.get('output_sha256') != file_digest(dataset_path)
            or manifest.get('signature') != data_signature(data)
            or manifest.get('held_out_digest') != held_out_digest(data)):
        raise ValueError('Prepared DuoRec data or mapping does not match its preprocessing manifest.')
    _use_read(read_root)
    from recbole.utils import get_model, init_seed

    # ReAd checkpoints serialize Python configuration objects; trust the source.
    state = torch.load(checkpoint, map_location='cpu', weights_only=False)
    if not isinstance(state, dict) or not {'config', 'state_dict'} <= state.keys():
        raise ValueError('Expected a ReAd DuoRec checkpoint with saved configuration.')
    config = state['config']
    if config['model'] != 'DuoRec':
        raise ValueError('Use a bare DuoRec checkpoint, not a TTARARec/ReAd checkpoint.')
    checkpoint_dataset = config['dataset']
    expected_name = ('Amazon_Office' if checkpoint_dataset == 'Amazon_Office_Products'
                     else checkpoint_dataset)
    if manifest.get('dataset') != expected_name:
        raise ValueError('Prepared dataset name differs from the DuoRec checkpoint.')
    saved = manifest.get('preprocessing_config')
    if not isinstance(saved, dict) or not saved:
        raise ValueError('Missing preprocessing settings in prepared-data manifest.')
    # Compare data settings, not training/model parameters or runtime paths.
    for key, value in saved.items():
        if key in ('seed', 'reproducibility'):
            continue
        if json.loads(json.dumps(config[key])) != value:
            raise ValueError(f'Prepared data used a different checkpoint setting: {key}')
    config['device'] = torch.device(device)
    config['use_gpu'] = torch.device(device).type == 'cuda'
    init_seed(config['seed'], config['reproducibility'])
    model = get_model('DuoRec')(config, _Catalogue(data, config))
    model.load_state_dict(state['state_dict'], strict=True)
    if any(not bool(torch.isfinite(p).all()) for p in model.parameters()):
        raise ValueError('Checkpoint contains nonfinite model weights.')
    adapter = RecBoleBackboneAdapter(model, data.num_items, model.max_seq_length, device)
    return ReadBackbone(adapter, int(config['seed']), checkpoint_dataset)
