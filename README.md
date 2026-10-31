# ELGAR

Official implementation of **ELGAR: Reusing Local Cases and Global Predictive Statistics for Sequential Recommendation**.

ELGAR (Explicit Local and Global Augmentation for Recommendation) is a plug-and-play framework for enhancing frozen sequential recommenders.
It combines source-deduplicated local case retrieval with global predictive statistics estimated from causal training prefixes, providing complementary score corrections through fixed-coefficient fusion.
Both branches reuse training interactions without backbone fine-tuning, hidden representations, or additional neural optimization.

## Layout

```text
ELGAR/
├── elgar/                  # Core retrieval, memory, fusion and evaluation
├── backbone/               # DuoRec, SASRec and FAVE inference adapters
│   ├── README.md           # Source, checkpoint and data acquisition
│   ├── SASRec/             # Bundled model, adapter and license
│   ├── DuoRec/             # ReAd adapter and acquisition guide
│   └── FAVE/               # Official FAVE adapter and acquisition guide
├── dataset/                # Dataset sources, layout and preparation guide
├── config/default.json     # The sole default configuration: (1, 0.5, 0.5)
├── constraints-reproduce.txt # Validated dependency versions
└── pyproject.toml          # Installation and dependency declarations
```

No datasets, checkpoints, generated caches, experiment analysis scripts or historical results are bundled. See [backbone/README.md](backbone/README.md) for acquiring backbone implementations and matching weights.
See [dataset/README.md](dataset/README.md) for the data sources and layout: Office, Beauty, Sports and Home use RecBole data; ML-100K and Steam use FAVE data.

## Installation

The full workflow was validated on Python 3.10.21, PyTorch 2.7.1+cu128, torchvision 0.22.1. Python 3.10 or 3.11 is recommended for ReAd's pandas 1.x dependency. 
Run these commands from the project root:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip setuptools wheel

# CUDA 12.8 build used in validation.
pip install torch==2.7.1 torchvision==0.22.1 \
  --index-url https://download.pytorch.org/whl/cu128
pip install -c constraints-reproduce.txt -e '.[read,data,fave]'
```

For CPU-only use, replace the Torch index with
`https://download.pytorch.org/whl/cpu`. `constraints-reproduce.txt` records validated dependency versions; it is separate from the model configuration.
A minimal core install is `pip install -e .`. Optional extras are `read` (ReAd DuoRec/data preparation), `data` (Google Drive downloads), and `fave` (the official FAVE source dependencies).

ReAd's bundled RecBole fork is required for DuoRec checkpoints; a standalone `recbole` installation is not a substitute. Model-source paths are explicit.
The validated external revisions are:

| Project | Git commit |
| --- | --- |
| ReAd | `e88d78fb5f42a1c6d310c527d93cff26ab5a6637` |
| FAVE | `09e81fb7c2fdfaf54092f7d5227ac085b78cc15d` |

## Prepare data separately

Download/import and preprocess data **before** model evaluation. All commands are under `dataset/`; see [dataset/README.md](dataset/README.md) for the six main-table datasets and full commands. 
The backbone adapters do not download, filter, remap or split datasets. Every backbone reads a prepared `dataset.pkl`.

## Evaluate DuoRec from ReAd

After preparing the data and acquiring weights, run from the project root:

```bash
python -m elgar \
  --backbone duorec \
  --dataset dataset/processed/Amazon_Beauty/dataset.pkl \
  --read-root external/ReAd \
  --checkpoint external/ReAd/pretrainmodel/DuoRec/Amazon_Beauty/model.pth \
  --config config/default.json \
  --cache-dir cache/Amazon_Beauty/memory \
  --output outputs/Amazon_Beauty_duorec.json
```

The adapter loads ReAd's DuoRec implementation and frozen weights, using the already prepared catalogue. 
It checks data provenance and the checkpoint's saved data settings without reopening `.inter` files. Missing preparation fails explicitly. 
Office's `Amazon_Office_Products` checkpoint alias is handled by the separate data preparation command.

## Evaluate SASRec

SASRec uses the implementation from the original CoMER project, bundled in `backbone/SASRec/model.py`. 
It is independent of ReAd and uses native SASRec state dictionaries. Do not pass `--read-root` for SASRec:

```bash
python -m elgar \
  --backbone sasrec \
  --dataset dataset/processed/ml-100k/dataset.pkl \
  --checkpoint checkpoints/sasrec/model.pth \
  --config config/default.json \
  --cache-dir cache/ml-100k/memory \
  --output outputs/ml-100k_sasrec.json
```

Native checkpoints require matching `args.txt` and `data.manifest.json` beside the weights. 
The adapted and legacy SASRec architectures use different state dictionaries; preserve the architecture setting recorded in `args.txt`.

## Evaluate FAVE

Install the official model's additional dependencies first:

```bash
pip install -c constraints-reproduce.txt -e '.[fave]'
```

Use the compatible Torch/torchvision builds and constraints described above.

```bash
python -m elgar \
  --backbone fave \
  --dataset dataset/processed/ml-100k/dataset.pkl \
  --checkpoint 'external/Fave/best for ml-100k/pretrain_best_finetuned.pt' \
  --backbone-source external/Fave \
  --backbone-config external/Fave/config/ml-100k-stage1-stage2.yaml \
  --config config/default.json \
  --cache-dir cache/ml-100k/memory \
  --output outputs/ml-100k_fave.json
```

Provide the native YAML matching the model architecture and training stage.
The adapter uses the official FAVE forward path and verifies that the supplied prepared-data manifest matches the native dataset name and mapping. 
FAVE source code and data can live in separate directories; no data are downloaded or processed by the adapter.

## Protocol and output

- Data preparation produces one validation item and one test item per user.
  Runtime loads those saved splits; it never generates them again.
- Native pickle input contains `train`, `val`, `test` and `smap` dictionaries.
  Each training sequence has at least two items; validation/test have one each.
  Item IDs must be contiguous; internal ID 0 is reserved for padding.
- Full-catalogue ranking with deterministic item-ID tie breaking. HR and NDCG
  are reported at 5, 10 and 20 for both the frozen backbone and ELGAR.
- All three backbones use unmasked real-item catalogues. The report records
  this candidate policy.
- ReAd DuoRec queries use the native maximum-length suffix for all scoring branches,
  matching the original integration. Memory fitting and retrieval indexing
  still use complete training sequences. Native pickle queries use their full
  observed histories; each backbone truncates its own input.
- Retrieval selects at most one case per source user and excludes the query
  user's cases. Global memory includes all legal training prefixes.
- `--split val` evaluates validation queries. `--device cpu|cuda|auto` and
  `--batch-size` override runtime settings. Default device selection is automatic.
- The default inference batch size is **512**, matching the validated results.
  Keep batch size, seed and device fixed when reproducing FAVE: its backbone
  ranks can change slightly with batch size. `--batch-size` can reduce memory
  usage; changing it constitutes a different inference setting.
- `--no-retrieval` and `--no-memory` disable the corresponding branches.
- Memory caches are validated against training content and item count; use a
  separate `--cache-dir` per dataset. Dense memory fitting requires quadratic
  space in the number of items. Lower `--batch-size` for scoring if needed.
- ReAd checkpoints contain serialized Python configuration objects. Only load
  checkpoints from a trusted source. Native dataset pickles accept plain Python
  containers only.

The console prints a compact accuracy table; `--output` saves metrics, effective configuration, candidate policy and input hashes as JSON. The module/installed entry points `python -m elgar` and `elgar-evaluate` accept the same options.

## Use an already loaded backbone

```python
from backbone.DuoRec import RecBoleBackboneAdapter
from elgar import Config, Recommender

# model is an already loaded ReAd DuoRec; data is a DatasetBundle with the
# exact same item mapping. histories contain observed item IDs only.
adapter = RecBoleBackboneAdapter(model, data.num_items, model.max_seq_length, 'cpu')
recommender = Recommender(data, None, Config(), backbone_adapter=adapter,
                          cache_dir='cache/my_dataset/memory', device='cpu')
scores = recommender.predict(histories, query_users)
# scores[:, 1:] corresponds to real item IDs 1..N; column 0 is padding.
```

## Reproducibility

Fresh official clones, new downloads and new caches have been exercised for DuoRec on Office/Beauty and FAVE on ML-100K. With the documented versions, seed and batch size, backbone and ELGAR results match the corresponding original results in all six HR/NDCG metrics.

## License and acknowledgments

ELGAR-specific code is released under the MIT License. The bundled SASRec model retains Apache-2.0 terms. ReAd, RecBole, DuoRec and FAVE retain their upstream licenses and should be cited when used; see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
