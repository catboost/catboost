# rotten_tomatoes

{% include [datasets-dataset__rotten_tomatoes](../_includes/work_src/reusage-python/dataset__rotten_tomatoes.md) %}


This dataset is best suited for text classification.

The training dataset contains 32712 objects. Each object is described by 22 columns of numerical, categorical and text features. The label column is not precisely specified. The dataset contains movie reviews. Every object in the dataset represents a unique review from a user and movie-specific information (such as synopsis, rating, etc.).

The validation dataset contains 8179 objects. The structure is identical to the training dataset.

## {{ dl--invoke-format }} {#method-call}

```python
rotten_tomatoes()
```

## {{ dl--output-format }} {#type-of-return-value}

{% include [datasets-datasets__output](../_includes/work_src/reusage-python/datasets__output.md) %}


## {{ dl__usage-examples }} {#usage-examples}

```python
from catboost.datasets import rotten_tomatoes

rotten_tomatoes_train, rotten_tomatoes_test = rotten_tomatoes()

print(rotten_tomatoes_train.head(3))

```

{% include [reusage-common-phrases-example-output](../_includes/work_src/reusage-common-phrases/example-output.md) %}


```bash
       id                                           synopsis rating_MPAA                                              genre  ...                 publisher        date    date_int rating_10
0   830.0  A gay New Yorker stages a marriage of convenie...           R  Art House and International | Comedy | Drama |...  ...  Las Vegas Review-Journal  2004-04-16  20040416.0       8.0
1  1161.0  Screenwriter Nimrod Antal makes an impressive ...           R  Action and Adventure | Art House and Internati...  ...                 E! Online  2005-04-22  20050422.0       6.0
2   596.0  "Arctic Tale" is an epic adventure that explor...           G                     Documentary | Special Interest  ...       New York Daily News  2007-07-27  20070727.0       6.0

[3 rows x 22 columns]
```

