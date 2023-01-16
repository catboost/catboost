# adult

{% include [datasets-datasets__adult__uci](../_includes/work_src/reusage-python/datasets__adult__uci.md) %}

This dataset is best suited for binary classification.
The training dataset contains 32561 objects. Each object is described by 15 columns of numerical and categorical features. The label column is not precisely specified.

The validation dataset contains 16281 objects. The structure is identical to the training dataset.

## {{ dl--invoke-format }} {#method-call}

```python
adult()
```

## {{ dl--output-format }} {#type-of-return-value}

{% include [datasets-datasets__output](../_includes/work_src/reusage-python/datasets__output.md) %}


## {{ dl__usage-examples }} {#usage-examples}

```python
from catboost.datasets import adult
adult_train, adult_test = adult()

print(adult_train.head(3))
```

{% include [reusage-common-phrases-example-output](../_includes/work_src/reusage-common-phrases/example-output.md) %}


```bash
    age         workclass    fnlwgt  education  education-num      marital-status         occupation   relationship   race   sex  capital-gain  capital-loss  hours-per-week native-country income
0  39.0         State-gov   77516.0  Bachelors           13.0       Never-married       Adm-clerical  Not-in-family  White  Male        2174.0           0.0            40.0  United-States  <=50K
1  50.0  Self-emp-not-inc   83311.0  Bachelors           13.0  Married-civ-spouse    Exec-managerial        Husband  White  Male           0.0           0.0            13.0  United-States  <=50K
2  38.0           Private  215646.0    HS-grad            9.0            Divorced  Handlers-cleaners  Not-in-family  White  Male           0.0           0.0            40.0  United-States  <=50K
```

