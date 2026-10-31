# Third-party notices

## SASRec

`backbone/SASRec/model.py` derives from
[pmixer/SASRec.pytorch](https://github.com/pmixer/SASRec.pytorch), based on
[the original SASRec implementation](https://github.com/kang205/SASRec).
The file retains both the legacy architecture and the adapted Transformer
variant from this project's preceding implementation. It is distributed under
Apache-2.0; the full license is in `backbone/SASRec/LICENSE.txt`.
ELGAR uses this model for inference; upstream training scripts are not bundled.

## ReAd, RecBole and DuoRec

The DuoRec adapter loads the implementation
from an external [ReAd checkout](https://github.com/xingt-tang/ReAd), which
bundles a modified [RecBole](https://github.com/RUCAIBox/RecBole).
ReAd and RecBole code, datasets and weights are not redistributed here.
Their upstream MIT licenses and attribution remain applicable to those downloads.
The original DuoRec project is https://github.com/RuihongQiu/DuoRec.

## FAVE

The FAVE adapter loads source from an external
[official FAVE checkout](https://github.com/Blue130/Fave).
FAVE source, datasets and weights are not redistributed. Consult the upstream
repository for its own license and checkpoint/data terms.

ELGAR adapters do not relicense externally obtained data, models or software.
