# amazon

{% include [datasets-datasets__amazon__purpose-desc](../_includes/work_src/reusage-python/datasets__amazon__purpose-desc.md) %}


This dataset is best suited for binary classification.

The training dataset contains 32769 objects. Each object is described by 10 columns of numerical features. The `ACTION` column is used as the label.

The validation dataset contains 58921 objects. The structure is identical to the training dataset with the following variations: 
- The `ACTION` column is omitted.
- The `id` column is added.

## {{ dl--invoke-format }} {#call-format}

```python
amazon()
```

## {{ dl--output-format }} {#parameters}

{% include [datasets-datasets__output](../_includes/work_src/reusage-python/datasets__output.md) %}


The train dataset contains the <q>ACTION</q> label.

## {{ dl__usage-examples }} {#usage-examples}

```python
from catboost.datasets import amazon
amazon_train, amazon_test = amazon()

print(amazon_train.head(3))
```

{% include [reusage-common-phrases-example-output](../_includes/work_src/reusage-common-phrases/example-output.md) %}


```bash
   ACTION  RESOURCE  MGR_ID  ROLE_ROLLUP_1  ROLE_ROLLUP_2  ROLE_DEPTNAME  ROLE_TITLE  ROLE_FAMILY_DESC  ROLE_FAMILY  ROLE_CODE
0       1     39353   85475         117961         118300         123472      117905            117906       290919     117908
1       1     17183    1540         117961         118343         123125      118536            118536       308574     118539
2       1     36724   14457         118219         118220         117884      117879            267952        19721     11788
```

