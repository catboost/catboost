# msrank

{% include [datasets-datasets__msrank](../_includes/work_src/reusage-python/datasets__msrank.md) %}


The training dataset contains 723412 objects. Each object is described by 138 columns. The first column contains the label value, the second one contains the identifier of the object's group (`{{ cd-file__col-type__GroupId }}`). All other columns contain features.

The validation dataset contains 241521 objects. The structure is identical to the training dataset.

## {{ dl--invoke-format }} {#method-call}

```python
msrank()
```

## {{ dl--output-format }} {#type-of-return-value}

{% include [datasets-datasets__output](../_includes/work_src/reusage-python/datasets__output.md) %}


## {{ dl__usage-examples }} {#usage-examples}

```python
from catboost.datasets import msrank
msrank_train, msrank_test = msrank()

print(msrank_train.head(3))

```

{% include [reusage-common-phrases-example-output](../_includes/work_src/reusage-common-phrases/example-output.md) %}


```bash
   0    1    2    3    4    5    6    7    8    9    10   11     12   13    14   ...        123        124        125        126  127   128         129   130      131      132   133   134  135  136  137
0  2.0  1.0  3.0  3.0  0.0  0.0  3.0  1.0  1.0  0.0  0.0  1.0  156.0  4.0   0.0  ...  -4.474452 -23.634899 -28.119826 -13.581932  3.0  62.0  11089534.0   2.0    116.0  64034.0  13.0   3.0  0.0  0.0  0.0
1  0.0  1.0  3.0  0.0  3.0  0.0  3.0  1.0  0.0  1.0  0.0  1.0  168.0  3.0  10.0  ... -24.041386  -7.222766 -28.119826 -12.483964  2.0  44.0         5.0  30.0  23836.0  63634.0   2.0   4.0  0.0  0.0  0.0
2  0.0  1.0  3.0  0.0  3.0  0.0  3.0  1.0  0.0  1.0  0.0  1.0  674.0  1.0   4.0  ... -24.041386  -4.474536 -28.119826 -15.288797  3.0  59.0         5.0   8.0    213.0  48469.0   1.0  13.0  0.0  0.0  0.0

[3 rows x 138 columns]

```

