# Data download and preprocessing

All acquisition and preprocessing commands live in this directory. Backbones
only load prepared data and model weights; evaluation never downloads data,
reads raw interaction tables, filters interactions or generates new splits.
Run commands from the **ELGAR project root** using `python -m dataset.<command>`.

## Main-table data sources

| Paper name | Identifier | Processed source |
| --- | --- | --- |
| Office | `Amazon_Office` | RecBole-format data used with ReAd |
| Beauty | `Amazon_Beauty` | RecBole-format data used with ReAd |
| Sports | `Amazon_Sports_and_Outdoors` | RecBole-format data used with ReAd |
| Home | `Amazon_Home` | RecBole-format data used with ReAd |
| ML-100K | `ml-100k` | Official FAVE data |
| Steam | `steam` | Official FAVE data |

The MovieLens identifier is `ml-100k`. Main-table Beauty uses the RecBole
source, even though FAVE also supplies a similarly named dataset.

## Layout and responsibilities

```text
dataset/
├── download.py             # Download a single official file or import a local copy
├── prepare.py              # Explicit preprocessing/export CLI
├── preprocess.py           # RecBole filtering/splits and FAVE validation
├── recbole/<name>/<name>.inter          # Downloaded RecBole interaction tables
├── fave/<name>/dataset.pkl              # Downloaded official FAVE pickles
└── processed/<name>/
    ├── dataset.pkl                     # Prepared train/val/test and ID mappings
    └── preprocessing.json              # Source/configuration hashes and provenance
```

Only source code and documentation are distributed. Downloaded files and
prepared outputs are ignored by Git. `--root` changes the data root for both
commands, and `--input` overrides a preparation input file. Identical files
are reused; replacing different data requires `--overwrite`.

## 1. Download or import data

### RecBole: Office, Beauty, Sports, Home

The default links are the official Amazon_ratings ZIP files listed in
[ReAd's linked RecBole data folder](https://github.com/xingt-tang/ReAd#datasets).
Download directly through the unified entry point:

```bash
for name in Office Beauty Sports Home; do
  python -m dataset.download --source recbole --name "$name"
done
```

`Office` maps to `Amazon_Office_Products` in the original ZIP, while `Home`
maps to `Amazon_Home_and_Kitchen`. Only the matching `.inter` member is streamed
to the canonical destination; `.item` files are not needed by this workflow.
The downloader never extracts arbitrary archive paths. `--url` overrides an
individual file link if upstream hosting changes. Google Drive folder URLs
are not individual files.

Already downloaded files can be imported without any network request:

```bash
for name in Amazon_Office Amazon_Beauty Amazon_Sports_and_Outdoors Amazon_Home; do
  python -m dataset.download --source recbole --name "$name" \
    --local-file "external/ReAd/recbole/dataset/$name/$name.inter"
done
```

Files retain their original bytes. The required format is tab-separated typed
columns, including `user_id`, `item_id` and `timestamp`; the saved RecBole
configuration determines any additional loaded fields, such as `rating`.

### FAVE: ML-100K and Steam

The default FAVE URLs are pinned to validated commit
`09e81fb7c2fdfaf54092f7d5227ac085b78cc15d` and point to the original pickles in
[Blue130/Fave](https://github.com/Blue130/Fave):

```bash
python -m dataset.download --source fave --name ml-100k
python -m dataset.download --source fave --name steam
```

Use `--revision` to select another FAVE Git revision. `--sha256` checks the
installed `.inter`/`.pkl` content, after ZIP extraction if applicable.
To import from an official checkout instead:

```bash
python -m dataset.download --source fave --name ml-100k --fave-root external/Fave
python -m dataset.download --source fave --name steam --fave-root external/Fave
```

A `download.json` records each source and SHA256. The downloader validates the
file format before installing it atomically; a failed download does not
replace an existing dataset.

## 2. Explicitly prepare data

### RecBole processing and native split export

Install `pip install -c constraints-reproduce.txt -e '.[read]'` and acquire ReAd's implementation and
checkpoints using [the DuoRec guide](../backbone/DuoRec/README.md).
Then process the four interaction tables with their saved checkpoint settings:

```bash
for name in Amazon_Office Amazon_Beauty Amazon_Sports_and_Outdoors Amazon_Home; do
  checkpoint_name="$name"
  if [ "$name" = Amazon_Office ]; then checkpoint_name=Amazon_Office_Products; fi
  python -m dataset.prepare --source recbole --name "$name" \
    --read-root external/ReAd \
    --checkpoint "external/ReAd/pretrainmodel/DuoRec/$checkpoint_name/model.pth"
done
```

This is the only stage that invokes RecBole data processing. It reproduces the
saved filtering and token mapping, orders each user's events by timestamp,
holds out the last event for test and the preceding event for validation,
and exports full training histories plus the exact item/user mappings.
No model is trained and no contrastive augmentation cache is constructed.

Before training new models, `--read-config external/ReAd/duorec.yaml` can replace
`--checkpoint`; the selected settings are recorded in `preprocessing.json`.
To reproduce released models, use their saved checkpoint configuration.

The exported pickle can also be used to train our bundled SASRec. Its weights
and checkpoint-side manifest must correspond to this same mapping and split;
preparing data does not make unrelated weights compatible automatically.

### FAVE validation without resplitting

```bash
python -m dataset.prepare --source fave --name ml-100k
python -m dataset.prepare --source fave --name steam
```

These commands validate `train`, `val`, `test` and `smap`, then copy the
original pickle **byte-for-byte** to `processed/`. They do not repeat upstream
filtering, remap IDs on disk, or replace the official splits. The native loader
only normalizes padding conventions in memory when reading zero-based IDs.

## 3. Load prepared data for evaluation

All three backbones use `--dataset dataset/processed/<name>/dataset.pkl`.
DuoRec and FAVE additionally verify the adjacent `preprocessing.json`; missing
preparation is an error, never a trigger to preprocess automatically.

```bash
python -m elgar --backbone duorec \
  --dataset dataset/processed/Amazon_Beauty/dataset.pkl \
  --read-root external/ReAd \
  --checkpoint external/ReAd/pretrainmodel/DuoRec/Amazon_Beauty/model.pth \
  --config config/default.json \
  --cache-dir cache/Amazon_Beauty/memory \
  --output outputs/Amazon_Beauty_duorec.json
```

SASRec and FAVE commands are in [the main README](../README.md).
The FAVE source checkout supplies model code only; it no longer needs a second
copy of the data under its own `datasets/` directory.

Runtime query tensors may truncate observed histories to a backbone's maximum
sequence length; this is input formatting, not a new train/validation/test
split. Retrieval indexing and global memory fitting use only prepared training
histories. All models use the complete real-item catalogue without sampled
negatives or consumed-item masking.

Dataset providers retain their original data terms. Download locations may
change upstream; explicit `--url` and `--local-file` remain available.
