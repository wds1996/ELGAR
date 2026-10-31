# DuoRec: ReAd implementation and released weights

Run all shell commands from the **ELGAR project root**, not this directory.
See [the main README](../../README.md) for evaluation commands.

Use the DuoRec implementation bundled by
[ReAd](https://github.com/xingt-tang/ReAd), rather than substituting a different
RecBole release or a different DuoRec reimplementation.

```bash
mkdir -p external
git clone https://github.com/xingt-tang/ReAd.git external/ReAd
git -C external/ReAd checkout e88d78fb5f42a1c6d310c527d93cff26ab5a6637
pip install -c constraints-reproduce.txt -e '.[read]'
```

1. Download the official checkpoint archive from
   [ReAd Releases](https://github.com/xingt-tang/ReAd/releases).
2. Extract it at the **ReAd checkout root** so the weights are located at
   `external/ReAd/pretrainmodel/DuoRec/<dataset>/model.pth`.
3. Acquire and prepare data using [dataset/README.md](../../dataset/README.md).
   Downloading, filtering and split generation are separate explicit commands.
4. Verify checksums using ReAd's `CHECKPOINTS.sha256` when supplied.

Implementation: `external/ReAd/recbole/model/sequential_recommender/duorec.py`.
ELGAR adapter: `adapter.py` and `recbole.py`.
The adapter only loads the frozen model against an already prepared catalogue
and checks `preprocessing.json`. It never reads `.inter` files or invokes
RecBole data processing, downloading, filtering or split generation.

```python
from elgar.data import load_dataset
from backbone.DuoRec import load_duorec

data_path = 'dataset/processed/Amazon_Beauty/dataset.pkl'
data = load_dataset(data_path)
loaded = load_duorec('external/ReAd/pretrainmodel/DuoRec/Amazon_Beauty/model.pth',
                     'external/ReAd', data, dataset_path=data_path)
adapter = loaded.adapter
```

Office weights use directory `Amazon_Office_Products`; the prepared dataset is
`Amazon_Office`. Use bare `DuoRec` checkpoints. `pretrainmodel/TTARARec` contains
ReAd's adapted models, not the frozen DuoRec checkpoint required here.

To train a fresh backbone, follow ReAd's environment/data setup and run from
its root:

```bash
python run_basic.py --model DuoRec --dataset Amazon_Beauty --config_files duorec.yaml
```

Keep the resulting checkpoint's saved config. Dataset availability and release
contents are controlled by upstream; ELGAR does not replace damaged weights
with a different local model automatically.

