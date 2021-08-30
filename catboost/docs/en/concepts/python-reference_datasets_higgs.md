# higgs

{% include [datasets-datasets__higgs](../_includes/work_src/reusage-python/datasets__higgs.md) %}


This dataset is best suited for binary classification.

The training dataset contains 10500000 objects. Each object is described by 29 columns. The first column contains the label value, all other columns contain numerical features.

The validation dataset contains 5000000 objects. The structure is identical to the training dataset.

## {{ dl--invoke-format }} {#method-call}

```python
higgs()
```

## {{ dl--output-format }} {#type-of-return-value}

{% include [datasets-datasets__output](../_includes/work_src/reusage-python/datasets__output.md) %}


## {{ dl__usage-examples }} {#usage-examples}

```python
from catboost.datasets import higgs
higgs_train, higgs_test = higgs()

print(higgs_train.head(3))

```

{% include [reusage-common-phrases-example-output](../_includes/work_src/reusage-common-phrases/example-output.md) %}


```bash
    0         1         2         3         4         5         6         7         8         9   ...        19        20        21        22        23        24        25        26        27        28
0  1.0  0.869293 -0.635082  0.225690  0.327470 -0.689993  0.754202 -0.248573 -1.092064  0.000000  ... -0.010455 -0.045767  3.101961  1.353760  0.979563  0.978076  0.920005  0.721657  0.988751  0.876678
1  1.0  0.907542  0.329147  0.359412  1.497970 -0.313010  1.095531 -0.557525 -1.588230  2.173076  ... -1.138930 -0.000819  0.000000  0.302220  0.833048  0.985700  0.978098  0.779732  0.992356  0.798343
2  1.0  0.798835  1.470639 -1.635975  0.453773  0.425629  1.104875  1.282322  1.381664  0.000000  ...  1.128848  0.900461  0.000000  0.909753  1.108330  0.985692  0.951331  0.803252  0.865924  0.780118

[3 rows x 29 columns]

```

