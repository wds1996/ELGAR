# Backbone adapters

Each backbone has its own package and acquisition instructions:

| Backbone | Implementation | Documentation |
| --- | --- | --- |
| SASRec | Bundled original-project implementation and native checkpoint adapter | [SASRec/README.md](SASRec/README.md) |
| DuoRec | ReAd's external DuoRec implementation and released checkpoints | [DuoRec/README.md](DuoRec/README.md) |
| FAVE | External official FAVE implementation and native checkpoints | [FAVE/README.md](FAVE/README.md) |

```text
backbone/
├── SASRec/
│   ├── model.py           # Original-project SASRec implementation
│   ├── adapter.py         # Native checkpoint adapter
│   ├── LICENSE.txt
│   └── README.md
├── DuoRec/
│   ├── adapter.py         # Public ReAd DuoRec loader
│   ├── recbole.py         # Frozen sequential-model scorer
│   └── README.md
└── FAVE/
    ├── adapter.py         # Official FAVE inference adapter
    └── README.md
```

All three directories are Python packages. Public imports are
`backbone.SASRec.SASRecAdapter`, `backbone.DuoRec.load_duorec` and
`backbone.FAVE.FaveAdapter`. The core factory in `backbone/__init__.py`
resolves these packages. No datasets or pretrained weights are bundled.
External source and asset paths are passed explicitly at runtime, so they do
not depend on where the adapter modules are located.

## Common adapter interface

```python
scores = adapter.score(histories)  # torch.Tensor, shape [batch, num_real_items]
```

- Histories contain nonempty integer sequences with IDs `1..N`; zero is padding.
- The adapter truncates to the model's native maximum sequence length.
- SASRec/FAVE use left padding; ReAd DuoRec uses right padding
  and explicit sequence lengths.
- Column `j` scores real item ID `j + 1`. All models are in evaluation mode
  with gradients disabled; ELGAR never updates their parameters.
- To bring your own loaded ReAd DuoRec, wrap it in `RecBoleBackboneAdapter`.

All download and preprocessing code lives in [dataset/](../dataset/README.md).
Backbones only consume prepared datasets and weights. Model-source paths
remain explicit; adapters never prepare missing inputs automatically.
