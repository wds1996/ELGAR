"""Prepare aligned native split pickles and record preprocessing provenance."""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path
import tempfile

import torch

from elgar.utils import data_signature, file_digest, held_out_digest, write_json
from .download import install_file
from .preprocess import canonical_name, prepare_recbole, use_read, validate_native_pickle


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', choices=('recbole', 'fave'), required=True)
    parser.add_argument('--name', required=True)
    parser.add_argument('--root', type=Path, default=Path('dataset'))
    parser.add_argument('--input', type=Path, help='Source file override; defaults to downloaded source layout.')
    parser.add_argument('--read-root', type=Path, help='ReAd implementation used for RecBole data.')
    setting = parser.add_mutually_exclusive_group()
    setting.add_argument('--checkpoint', type=Path, help='Trusted ReAd DuoRec checkpoint with saved preprocessing config.')
    setting.add_argument('--read-config', type=Path, help='ReAd DuoRec YAML for preparing data before training.')
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args(argv)
    name = canonical_name(args.source, args.name)
    filename = f'{name}.inter' if args.source == 'recbole' else 'dataset.pkl'
    incoming = args.input or args.root / args.source / name / filename
    output = args.root / 'processed' / name
    checkpoint_hash = config_hash = None
    config_summary = None
    if args.source == 'recbole':
        if args.read_root is None or not (args.checkpoint or args.read_config):
            parser.error('RecBole preparation requires --read-root and either --checkpoint or --read-config.')
        use_read(args.read_root)
        from recbole.utils import get_model
        if args.checkpoint:
            state = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
            config = state['config']
            if config['model'] != 'DuoRec':
                raise ValueError('Use a bare ReAd DuoRec checkpoint for the saved preprocessing settings.')
            checkpoint_hash = file_digest(args.checkpoint)
            del state
        else:
            from recbole.config import Config
            config = Config(model=get_model('DuoRec'), dataset=name,
                            config_file_list=[str(args.read_config.resolve())],
                            config_dict={'data_path': str(incoming.resolve().parent.parent)})
            config_hash = file_digest(args.read_config)
        config['device'] = torch.device('cpu')
        config['use_gpu'] = False
        # Only the file is staged if a caller supplies a differently named
        # source; RecBole still sees its canonical <dataset>.inter filename.
        with tempfile.TemporaryDirectory(prefix='elgar-preprocess-') as tmp:
            folder = Path(tmp)
            install_file(incoming, folder / filename)
            prepared = prepare_recbole(config, folder, name)
            raw = prepared.data.raw
        config_summary = {key: config[key] for key in (
            'seed', 'reproducibility', 'eval_setting', 'MAX_ITEM_LIST_LENGTH',
            'min_user_inter_num', 'min_item_inter_num', 'load_col',
            'USER_ID_FIELD', 'ITEM_ID_FIELD', 'TIME_FIELD', 'rm_dup_inter',
            'user_inter_num_interval', 'item_inter_num_interval')}
        with tempfile.TemporaryDirectory(prefix='elgar-native-export-') as tmp:
            native = Path(tmp) / 'dataset.pkl'
            with native.open('wb') as handle:
                pickle.dump(raw, handle, protocol=4)
            digest = install_file(native, output / 'dataset.pkl', overwrite=args.overwrite)
        data = validate_native_pickle(output / 'dataset.pkl')
        transformation = 'native RecBole filtering/token mapping; chronological last-two holdout; native pickle export'
    else:
        if args.read_root or args.checkpoint or args.read_config:
            parser.error('ReAd options apply only to RecBole data.')
        data = validate_native_pickle(incoming)
        digest = install_file(incoming, output / 'dataset.pkl', overwrite=args.overwrite)
        transformation = 'byte-identical copy of official FAVE pickle; no resplitting or remapping on disk'
    # This describes prepared data. It is NOT a claim that arbitrary backbone
    # weights were trained on it; checkpoint-side SASRec manifests remain required.
    write_json(output / 'preprocessing.json', {
        'source': args.source, 'dataset': name,
        'input_sha256': file_digest(incoming), 'output_sha256': digest,
        'transformation': transformation, 'checkpoint_sha256': checkpoint_hash,
        'read_config_sha256': config_hash, 'preprocessing_config': config_summary,
        'signature': data_signature(data), 'held_out_digest': held_out_digest(data),
        'users': len(data.users), 'items': data.num_items,
        'training_events': sum(len(seq) for seq in data.raw['train'].values())})
    print(f'[ready] {output / "dataset.pkl"}: {len(data.users)} users, {data.num_items} items')
    return output / 'dataset.pkl'


if __name__ == '__main__':
    main()
