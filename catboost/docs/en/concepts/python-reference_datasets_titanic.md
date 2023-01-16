# titanic

{% include [datasets-datasets__titanic__purpose-desc](../_includes/work_src/reusage-python/datasets__titanic__purpose-desc.md) %}


This dataset is best suited for binary classification.

The training dataset contains 891 objects. Each object is described by 12 columns of numerical and categorical features. The `Survived` column is often used as the label.

The validation dataset contains 418 objects. The structure is similar to the training dataset except for the `Survived` column which is omitted.

## {{ dl--invoke-format }} {#call-format}

```python
titanic()
```

## {{ dl--output-format }} {#output-format}

{% include [datasets-datasets__output](../_includes/work_src/reusage-python/datasets__output.md) %}


## {{ dl__usage-examples }} {#usage-examples}

```python
from catboost.datasets import titanic
titanic_train, titanic_test = titanic()

print(titanic_train.head(3))
```

{% include [reusage-common-phrases-example-output](../_includes/work_src/reusage-common-phrases/example-output.md) %}


```bash
   PassengerId  Survived  Pclass                                               Name     Sex   Age  SibSp  Parch            Ticket     Fare Cabin Embarked
0            1         0       3                            Braund, Mr. Owen Harris    male  22.0      1      0         A/5 21171   7.2500   NaN        S
1            2         1       1  Cumings, Mrs. John Bradley (Florence Briggs Th...  female  38.0      1      0          PC 17599  71.2833   C85        C
2            3         1       3                             Heikkinen, Miss. Laina  female  26.0      0      0  STON/O2. 3101282   7.9250   NaN        S
```

