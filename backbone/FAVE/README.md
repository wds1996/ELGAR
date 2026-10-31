# FAVE: official external implementation

Run all shell commands from the **ELGAR project root**, not this directory.
See [the main README](../../README.md) for evaluation commands.

Install `pip install -c constraints-reproduce.txt -e '.[fave]'` for the official source dependencies
(`timm`, including a compatible torchvision build).

Source and released artifacts:
[Blue130/Fave](https://github.com/Blue130/Fave).

```bash
mkdir -p external
# Install Git LFS first. Skip automatic downloads of unrelated checkpoints.
GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/Blue130/Fave.git external/Fave
GIT_LFS_SKIP_SMUDGE=1 git -C external/Fave checkout 09e81fb7c2fdfaf54092f7d5227ac085b78cc15d
git -C external/Fave lfs pull \
  --include='best for ml-100k/pretrain_best_finetuned.pt' --exclude=''
```

Follow upstream instructions for obtaining or training the two-stage weights.
Download and prepare datasets separately with
[the dataset commands](../../dataset/README.md). Expected model-source layout:

```text
external/Fave/
├── src/model.py
├── src/fave.py
├── config/<native-config>.yaml
└── best for <dataset>/
    ├── pretrain_best.pt
    └── pretrain_best_finetuned.pt
```

`adapter.py` loads `src/model.py` and `src/fave.py` from `--backbone-source` and
uses the exact YAML supplied with `--backbone-config`. Both source path and
YAML are required. It supports native state dictionaries and strips a leading
`module.` from DataParallel keys. The `_finetuned` filename suffix selects
stage two, as in the native release; preserve the original filenames.

The adapter only loads prepared data and checks the adjacent
`preprocessing.json` against the native YAML's dataset name, file hash and
mapping. It does not download, filter or split data. No duplicate dataset is
required inside the official FAVE source checkout. FAVE source code, weights
and datasets retain their upstream terms.

For the validated numbers, use seed 2026 and inference batch size 512
(the defaults). Keep the device and batch size fixed; smaller batches can
change the native FAVE backbone's ranks slightly.
