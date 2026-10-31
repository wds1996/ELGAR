"""Evaluate a frozen backbone and ELGAR with full-catalogue HR/NDCG."""
from __future__ import annotations
import argparse
from dataclasses import replace
from pathlib import Path
import numpy as np
import torch
from .config import Config, load_config
from .data import load_dataset, make_evaluation_queries
from .pipeline import Recommender, fuse
from .utils import file_digest, ranking_metrics, setup_runtime, write_json
from .progress import progress


def numpy_ranks(scores, targets):
    if isinstance(scores, torch.Tensor):
        if (scores.ndim != 2 or not len(scores) or scores.shape[1] < 2
                or scores.is_complex() or not bool(torch.isfinite(scores).all())):
            raise ValueError('Invalid catalogue scores/targets.')
        t = torch.as_tensor(targets, device=scores.device)
        if (t.shape != (len(scores),) or t.dtype.is_floating_point
                or t.dtype.is_complex or t.dtype == torch.bool):
            raise ValueError('Targets must be one integer ID per query.')
        t = t.long()
        if bool(((t < 1) | (t >= scores.shape[1])).any()):
            raise ValueError('Invalid target ID.')
        positive = scores.gather(1,t[:,None])
        real = scores[:,1:]
        ids = torch.arange(1,scores.shape[1],device=scores.device)[None,:]
        return (1+(real>positive).sum(1)+((real==positive)&(ids<t[:,None])).sum(1)).cpu().numpy()
    s, t = np.asarray(scores), np.asarray(targets)
    if (s.ndim != 2 or not len(s) or s.shape[1] < 2 or s.dtype.kind not in 'fiu'
            or t.shape != (len(s),) or t.dtype.kind not in 'iu' or not np.isfinite(s).all()):
        raise ValueError('Invalid catalogue scores/targets.')
    if np.any((t < 1) | (t >= s.shape[1])):
        raise ValueError('Invalid target ID.')
    target_scores = s[np.arange(len(s)), t]
    real = s[:, 1:]
    ids = np.arange(1, s.shape[1])[None, :]
    return 1 + (real > target_scores[:, None]).sum(1) + ((real == target_scores[:, None]) & (ids < t[:, None])).sum(1)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backbone', choices=('duorec', 'sasrec', 'fave'), required=True)
    parser.add_argument('--checkpoint', required=True, type=Path)
    parser.add_argument('--config', type=Path, help='Optional ELGAR JSON; defaults to Config().')
    parser.add_argument('--dataset', required=True, type=Path, help='Already prepared dataset.pkl for any backbone.')
    parser.add_argument('--read-root', type=Path, help='ReAd checkout for DuoRec only.')
    parser.add_argument('--backbone-source', type=Path, help='External official FAVE repository.')
    parser.add_argument('--backbone-config', type=Path, help='Native FAVE YAML configuration.')
    parser.add_argument('--device', choices=('auto', 'cpu', 'cuda'))
    parser.add_argument('--batch-size', type=int)
    parser.add_argument('--split', choices=('val', 'test'), default='test')
    parser.add_argument('--retrieval', action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument('--memory', action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument('--cache-dir', type=Path, default=Path('cache/memory'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    if args.backbone != 'duorec' and args.read_root:
        parser.error('--read-root applies only to DuoRec.')
    cfg = load_config(args.config) if args.config else Config()
    if args.device is not None:
        cfg = replace(cfg, device=args.device)
    if args.batch_size is not None:
        cfg = replace(cfg, inference_batch_size=args.batch_size)
    for branch in ('retrieval', 'memory'):
        enabled = getattr(args, branch)
        if enabled is not None:
            cfg = replace(cfg, **{branch: replace(getattr(cfg, branch), enabled=enabled)})
    device = setup_runtime(cfg.seed, cfg.threads, cfg.device)
    # Runtime only loads saved splits. Acquisition/filtering/export are
    # explicit commands under dataset/ and are never invoked here.
    data = load_dataset(args.dataset)
    queries = make_evaluation_queries(data, args.split)
    context = None
    if args.backbone == 'duorec':
        if args.read_root is None:
            parser.error('DuoRec requires --read-root pointing to a ReAd checkout.')
        from backbone.DuoRec import load_duorec
        context = load_duorec(args.checkpoint, args.read_root, data, device=device,
                             dataset_path=args.dataset)
        cfg = replace(cfg, seed=context.seed)
        # Native input formatting only: retain ReAd's scoring-time suffix for
        # all ELGAR branches, without changing the prepared train/val/test split.
        queries.histories = [h[-context.adapter.max_len:] for h in queries.histories]
    pipeline = Recommender(data, args.checkpoint, cfg, cache_dir=args.cache_dir,
                           device=device, backbone=args.backbone,
                           backbone_config=args.backbone_config,
                           backbone_source=args.backbone_source,
                           backbone_adapter=context.adapter if context else None)
    weights = {'backbone': np.array([1., 0., 0.]), 'ELGAR': pipeline.weights}
    ranks = {name: [] for name in weights}
    for start in progress(range(0, len(queries), cfg.inference_batch_size),
                          desc=f'[{args.split}]', unit='batch'):
        stop = min(start + cfg.inference_batch_size, len(queries))
        components = pipeline.components(queries.histories[start:stop], queries.users[start:stop],
                                         return_tensor=device.type == 'cuda')
        for name, coefficients in weights.items():
            scores = fuse(components, coefficients)
            ranks[name].append(numpy_ranks(scores, queries.targets[start:stop]))
    measured = {name: ranking_metrics(np.concatenate(parts)) for name, parts in ranks.items()}
    report = {'backbone': args.backbone, 'split': args.split,
              'checkpoint': str(args.checkpoint.resolve()),
              'checkpoint_sha256': file_digest(args.checkpoint),
              'dataset_source': str(args.dataset.resolve()),
              'dataset_sha256': data.source_sha256,
              'configuration': cfg.to_dict(), 'coefficients': pipeline.weights.tolist(),
              'candidate_policy': 'full catalogue, unmasked',
              'test_or_validation_queries': len(queries), 'metrics': measured}
    write_json(args.output, report)
    keys = [f'{metric}@{k}' for metric in ('HR', 'NDCG') for k in (5, 10, 20)]
    print(' | '.join(['Model', *keys]))
    for name, values in measured.items():
        print(' | '.join([name, *(f'{values[key]:.4f}' for key in keys)]))
    print(f'Report: {args.output}')
    return report


if __name__ == '__main__':
    main()
