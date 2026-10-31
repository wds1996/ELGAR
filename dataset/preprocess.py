"""Shared RecBole preprocessing and native-pickle validation.

No backbone weights are constructed or updated here. ReAd supplies the
RecBole implementation; ELGAR owns the data preparation entry points.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

from elgar.data import DatasetBundle, load_dataset
from elgar.utils import file_digest

RECBOLE_NAMES = ('Amazon_Office', 'Amazon_Beauty',
                 'Amazon_Sports_and_Outdoors', 'Amazon_Home')
FAVE_NAMES = ('ml-100k', 'steam')
ALIASES = {'Office': 'Amazon_Office', 'Beauty': 'Amazon_Beauty',
           'Sports': 'Amazon_Sports_and_Outdoors', 'Home': 'Amazon_Home',
           'ML-100K': 'ml-100k', 'Steam': 'steam',
           'Amazon_Office_Products': 'Amazon_Office',
           'Amazon_Home_and_Kitchen': 'Amazon_Home'}


def canonical_name(source, name):
    name = ALIASES.get(name, name)
    allowed = RECBOLE_NAMES if source == 'recbole' else FAVE_NAMES if source == 'fave' else ()
    if name not in allowed:
        raise ValueError(f'Unsupported {source} dataset: {name}; choose from {allowed}.')
    return name


def use_read(root):
    root = Path(root).resolve()
    package = root / 'recbole'
    if not (package / 'config/__init__.py').is_file():
        raise FileNotFoundError(f'ReAd RecBole implementation not found: {package}')
    loaded = sys.modules.get('recbole')
    if loaded is not None and Path(loaded.__file__).resolve().parent != package:
        raise RuntimeError('A different RecBole is already imported. Start a fresh '
                           'Python process and pass the ReAd repository as --read-root.')
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return root


@dataclass
class PreparedRecBole:
    data: DatasetBundle
    dataset: object
    source: Path


def prepare_recbole(config, folder, name):
    """Use native filtering/mapping, then chronological last-two holdout.

    Called only by the explicit preparation command. Backbone inference never
    calls this function. No training loaders or augmentation caches are created.
    """
    from recbole.config import EvalSetting
    from recbole.data import create_dataset
    from recbole.utils import init_seed

    folder = Path(folder).resolve()
    source = folder / f'{name}.inter'
    validate_interactions(source)
    config['dataset'] = name
    config['data_path'] = str(folder)
    setting = EvalSetting(config)
    if (setting.ordering_args.get('strategy') != 'by'
            or setting.ordering_args.get('field') != config['TIME_FIELD']
            or setting.ordering_args.get('ascending') is not True
            or setting.group_field != config['USER_ID_FIELD']
            or setting.split_args.get('strategy') != 'loo'
            or setting.split_args.get('leave_one_num') != 2):
        raise ValueError('Requires time-ordered user-level leave-two-out settings '
                         '(one validation item and one test item).')
    if config['benchmark_filename'] is not None:
        raise ValueError('Explicit benchmark files are not supported by this bridge.')
    init_seed(config['seed'], config['reproducibility'])
    dataset = create_dataset(config)
    dataset._change_feat_format()
    dataset.sort(by=[dataset.uid_field, dataset.time_field], ascending=True)
    feat = dataset.inter_feat
    whole = {}
    for user, item in zip(feat[dataset.uid_field].cpu().numpy(),
                          feat[dataset.iid_field].cpu().numpy()):
        whole.setdefault(int(user), []).append(int(item))
    if not whole or any(len(seq) < 4 for seq in whole.values()):
        raise ValueError('Each user needs at least two training events plus val/test.')
    n = dataset.item_num - 1
    users = sorted(whole)
    tokens = dataset.field2id_token[dataset.iid_field]
    raw = {'train': {u: seq[:-2] for u, seq in whole.items()},
           'val': {u: [whole[u][-2]] for u in users},
           'test': {u: [whole[u][-1]] for u in users},
           'smap': {str(tokens[i]): i for i in range(1, n + 1)}}
    # Retain original user tokens as well as exact item tokens in exports.
    user_tokens = dataset.field2id_token[dataset.uid_field]
    raw['umap'] = {str(user_tokens[u]): u for u in users}
    data = DatasetBundle(raw, users, n, source_sha256=file_digest(source))
    return PreparedRecBole(data, dataset, source)


def validate_interactions(path):
    path = Path(path)
    with path.open('r', encoding='utf-8-sig') as handle:
        columns = handle.readline().strip().split('\t')
    names = {column.split(':', 1)[0] for column in columns if ':' in column}
    if not {'user_id', 'item_id', 'timestamp'} <= names:
        raise ValueError(f'Expected a typed, tab-separated RecBole .inter file: {path}')


def validate_native_pickle(path):
    """Validate without reserializing or resplitting the official pickle."""
    return load_dataset(path)
