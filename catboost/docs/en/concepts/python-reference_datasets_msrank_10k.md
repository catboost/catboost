# msrank_10k

{% include [datasets-datasets__msrank_10k](../_includes/work_src/reusage-python/datasets__msrank_10k.md) %}


The training dataset contains 10000 objects. Each object is described by 138 columns. The first column contains the label value, the second one contains the identifier of the object's group (`{{ cd-file__col-type__GroupId }}`). All other columns contain features.

The validation dataset contains 10000 objects. The structure is identical to the training dataset.

## {{ dl--invoke-format }} {#method-call}

```python
msrank_10k()
```

## {{ dl--output-format }} {#type-of-return-value}

{% include [datasets-datasets__output](../_includes/work_src/reusage-python/datasets__output.md) %}


## {{ dl__usage-examples }} {#usage-examples}

```
from catboost.datasets import msrank_10k
msrank_10k_train, msrank_10k_test = msrank_10k()

print(msrank_10k_train.head(3))

```

{% include [reusage-common-phrases-example-output](../_includes/work_src/reusage-common-phrases/example-output.md) %}


```bash
   0      1    2    3    4    5    6    7    8         9   10   11   12   13   14   ...       123        124        125        126  127  128       129  130  131    132  133  134  135  136  137
0  2.0    1    3    3    0    0    3  1.0  1.0  0.000000  0.0  1.0  156    4    0  ...  -4.474452 -23.634899 -28.119826 -13.581932    3   62  11089534    2  116  64034   13    3    0    0  0.0
1  2.0    1    3    0    3    0    3  1.0  0.0  1.000000  0.0  1.0  406    0    5  ... -24.041386  -5.143860 -28.119826 -11.411068    2   54  11089534    2  124  64034    1    2    0    0  0.0
2  0.0    1    3    0    2    0    3  1.0  0.0  0.666667  0.0  1.0  146    0    3  ... -24.041386 -14.689844 -28.119826 -11.436378    3   45         3    1  124   3344   14   67    0    0  0.0

[3 rows x 138 columns]

```

