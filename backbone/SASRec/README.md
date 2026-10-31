# SASRec

Run all shell commands from the **ELGAR project root**, not this directory.
See [the main README](../../README.md) for evaluation commands.

SASRec uses the implementation from the original CoMER project, included in
`model.py`; it does not load SASRec from ReAd or RecBole. Use the
matching native checkpoints together with their matching data,
`args.txt` and `data.manifest.json`.

`model.py` retains the legacy and adapted models used by the original
CoMER project; `adapter.py` is their inference adapter. The adapted architecture
uses input LayerNorm, causal PreLN attention with padding masking, GELU
feed-forward layers and an item output bias. It derives from
[pmixer/SASRec.pytorch](https://github.com/pmixer/SASRec.pytorch) under Apache-2.0.
See also [the original SASRec](https://github.com/kang205/SASRec).
Training code is intentionally not bundled.

For native evaluation omit `--read-root` and supply `--dataset`. Required layout:

```text
checkpoints/sasrec/
├── model.pth
├── args.txt
└── data.manifest.json
```

`args.txt` is a comma-separated `key,value` file with `dataset`, `maxlen`,
`hidden_units`, `num_blocks`, `num_heads`, `dropout_rate`, `norm_first` and,
`architecture` and `padding_mask`. Use `architecture,pasted` for our adapted
implementation; `architecture,legacy` is reserved for legacy checkpoints. Keep these files from the
same original training run. ELGAR does not assume that an arbitrary upstream
state dictionary has the adapted model's architecture.

For a freshly prepared legacy checkpoint without a manifest, verify the data
mapping/splits used for training and create a manifest once:

```python
import json
from pathlib import Path
from elgar.data import load_dataset
from elgar.utils import data_signature, held_out_digest

data = load_dataset('dataset/processed/ml-100k/dataset.pkl')
manifest = {'schema': 'elgar-sasrec-data-v1',
            'signature': data_signature(data),
            'held_out_digest': held_out_digest(data),
            'num_users': len(data.users)}
Path('checkpoints/sasrec/data.manifest.json').write_text(json.dumps(manifest))
```

This records a mapping that **you have verified**; it cannot establish that an
unrelated checkpoint was trained on that mapping. Existing original-project
manifests remain accepted for compatibility. Native SASRec evaluation uses
all real items without a consumed-item mask.


Dataset acquisition and preprocessing are explicit commands under
[dataset/](../../dataset/README.md). SASRec only loads the resulting pickle
and its own model weights; it never downloads data or constructs new splits.

The implementation is already included here; no ReAd SASRec source is used.
Native weights must be supplied by a matching training run. This release does
not bundle a SASRec checkpoint or a backbone training runner, and does not
claim that weights from another SASRec implementation are interchangeable.
