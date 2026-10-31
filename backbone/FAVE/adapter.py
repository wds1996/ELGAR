"""Inference adapter for the externally supplied official FAVE implementation."""
from __future__ import annotations
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace
import torch
import yaml
from elgar.data import validate_histories
from elgar.utils import data_signature, held_out_digest


def _native_fave_class(root):
    source = root / 'src'
    for filename in ('fave.py', 'model.py'):
        if not (source / filename).is_file():
            raise FileNotFoundError(f'External Fave project is missing: {source / filename}')
    # Fave's model.py imports its sibling as "fave". Load both from the native
    # project without making that generic module name permanent in sys.modules.
    old_fave = sys.modules.get('fave')
    try:
        for name, filename in (('fave', 'fave.py'), ('elgar_external_fave_model', 'model.py')):
            spec = importlib.util.spec_from_file_location(name, source / filename)
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            spec.loader.exec_module(module)
        return module.Att_Fave_model, module.create_model_Fave
    finally:
        if old_fave is None:
            sys.modules.pop('fave', None)
        else:
            sys.modules['fave'] = old_fave


class FaveAdapter:
    """Run a native Fave checkpoint on the identical native item mapping."""

    def __init__(self, checkpoint, data, device='cpu', config_path=None, source_root=None):
        checkpoint = Path(checkpoint)
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        if config_path is None or source_root is None:
            raise ValueError('FAVE requires --backbone-config and --backbone-source.')
        source_root = Path(source_root).resolve()
        config_path = Path(config_path)
        if not config_path.is_file():
            raise FileNotFoundError(config_path)
        self.config_path = config_path
        native = yaml.safe_load(config_path.read_text(encoding='utf-8'))
        if not isinstance(native, dict) or not isinstance(native.get('dataset'), str):
            raise ValueError('Invalid native Fave YAML configuration.')
        if data.source_path is None:
            raise ValueError('FAVE requires a prepared dataset loaded from disk.')
        metadata = Path(data.source_path).with_name('preprocessing.json')
        if not metadata.is_file():
            raise FileNotFoundError(f'{metadata} is required; run dataset.prepare first.')
        manifest = json.loads(metadata.read_text())
        if (not isinstance(manifest, dict) or manifest.get('source') != 'fave'
                or manifest.get('dataset') != native['dataset']
                or manifest.get('output_sha256') != data.source_sha256
                or manifest.get('signature') != data_signature(data)
                or manifest.get('held_out_digest') != held_out_digest(data)):
            raise ValueError('Prepared FAVE dataset/configuration/mapping mismatch.')
        required = ('max_len', 'hidden_size', 'emb_dropout', 'dropout', 'mask_ratio',
                    'eps', 'train_mask_ratio', 'infer_mask_ratio', 'loss_aux_weight',
                    'loss_straight_weight', 'loss_pretrain_weight', 'mask_end',
                    'num_blocks', 'drop_path_rate', 'gamma_init', 'last',
                    'lambda_uncertainty', 'sample_N', 'eps_reverse', 'm_logNorm',
                    's_logNorm', 's_modsamp', 'sampling_method')
        missing = set(required) - native.keys()
        if missing:
            raise ValueError(f'Incomplete native Fave YAML: {sorted(missing)}')
        self.max_len = native['max_len']
        if type(self.max_len) is not int or self.max_len < 1:
            raise ValueError('Invalid native Fave sequence length.')
        self.num_items = data.num_items
        self.device = torch.device(device)
        native_args = SimpleNamespace(**native, item_num=self.num_items + 1)
        native_model, native_factory = _native_fave_class(source_root)
        model = native_model(native_factory(native_args), native_args)
        try:
            weights = torch.load(checkpoint, map_location='cpu', weights_only=True)
            if not isinstance(weights, dict):
                raise ValueError('Expected a native Fave state_dict.')
            weights = {key.removeprefix('module.'): value for key, value in weights.items()}
            model.load_state_dict(weights, strict=True)
        except (RuntimeError, ValueError, TypeError, KeyError) as exc:
            raise ValueError('Native Fave checkpoint and YAML are incompatible.') from exc
        if any(not bool(torch.isfinite(parameter).all()) for parameter in model.parameters()):
            raise ValueError('Native Fave checkpoint contains nonfinite weights.')
        model.stage = 'finetune' if checkpoint.stem.endswith('_finetuned') else 'pretrain'
        self.model = model.to(self.device).eval().requires_grad_(False)

    @torch.inference_mode()
    def score(self, histories) -> torch.Tensor:
        histories = validate_histories(histories, self.num_items)
        sequences = torch.zeros((len(histories), self.max_len), dtype=torch.long, device=self.device)
        for row, history in enumerate(histories):
            suffix = history[-self.max_len:]
            sequences[row, -len(suffix):] = torch.tensor(suffix.tolist(), device=self.device)
        # The native Fave forward path ignores tag when train_flag=False.
        _, representation, *_ = self.model(sequences, None, train_flag=False)
        return self.model.fave_rep_pre(representation)[:, 1:]


