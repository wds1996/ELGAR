"""Frozen DuoRec, SASRec and FAVE adapters for ELGAR.

All adapters score only real items, with columns corresponding to IDs 1..N.
"""


def load_backbone(name, checkpoint, data, device='cpu', config_path=None,
                  source_root=None):
    if name == 'sasrec':
        from .SASRec import SASRecAdapter
        return SASRecAdapter(checkpoint, data, device)
    if name == 'fave':
        from .FAVE import FaveAdapter
        return FaveAdapter(checkpoint, data, device, config_path, source_root)
    if name == 'duorec':
        raise ValueError('Use backbone.DuoRec.load_duorec for ReAd checkpoints, '
                         'or pass an already loaded RecBoleBackboneAdapter.')
    raise ValueError(f'Unsupported backbone: {name}')
