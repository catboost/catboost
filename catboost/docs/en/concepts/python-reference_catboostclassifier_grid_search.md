# grid_search

{% include [grid_search-grid-search__div__desc](../_includes/work_src/reusage-python/grid-search__div__desc.md) %}


## {{ dl--invoke-format }} {#method-call-format}

```python
grid_search(param_grid,
            X,
            y=None,
            cv=3,
            partition_random_seed=0,
            calc_cv_statistics=True,
            search_by_train_test_split=True,
            refit=True,
            shuffle=True,
            stratified=None,
            train_size=0.8,
            verbose=True,
            plot=False,
            log_cout=sys.stdout,
            log_cerr=sys.stderr)
```

## {{ dl--parameters }} {#parameters}

### param_grid

#### Description


Dictionary with parameters names ({{ python-type--string }}) as keys and lists of parameter settings to try as values, or a list of such dictionaries, in which case the grids spanned by each dictionary in the list are explored.

This enables searching over any sequence of parameter settings.

**Possible types**


- {{ python-type--dict }}
- {{ python-type--list }}

**Default value**

{{ python--required }}

### X

#### Description

The description is different for each group of possible types.

**Possible types**

{% cut "{{ python-type--pool }}" %}

The input training dataset.

{% note info %}

If a nontrivial value of the `cat_features` parameter is specified in the constructor of this class, {{ product }} checks the equivalence of categorical features indices specification from the constructor parameters and in this Pool class.

{% endnote %}

{% endcut %}

{% cut "{{ python-type--numpyarray }}, {{ python-type--pandasDataFrame }}" %}

The input training dataset in the form of a two-dimensional feature matrix.

{% endcut %}

{% cut "{{ python_type__pandas-SparseDataFrame }}, {{ python_type__scipy-sparse-spmatrix }} (all subclasses except dia_matrix)" %}

{% include [libsvm-libsvm__desc](../_includes/work_src/reusage-formats/libsvm__desc.md) %}

{% endcut %}

**Default value**

{{ python--required }}

### y

#### Description

{% include [methods-param-desc-label--short-desc1](../_includes/work_src/reusage/label--short-desc1.md) %}


{% include [methods-param-desc-label--short-desc2](../_includes/work_src/reusage/label--short-desc2.md) %}


{% note info %}

Do not use this parameter if the input training dataset (specified in the `X` parameter) type is {{ python-type--pool }}.

{% endnote %}

**Possible types**


- {{ python-type--numpyarray }}
- {{ python-type--pandasSeries }}

**Default value**

{{ python-type--none }}

### cv

#### Description


The cross-validation splitting strategy.

The interpretation of this parameter depends on the input data type:
- None — Use the default three-fold cross-validation.
- {{ python-type--int }} — The number of folds in a (Stratified)KFold
- {% include [reusage-python-object-scikitlearn](../_includes/work_src/reusage-python/object-scikitlearn.md) %}

- An iterable yielding train and test splits as arrays of indices.

**Possible types**


- {{ python-type--int }}
- scikit-learn splitter {{ python-type__object }}
- cross-validation generator
- iterable

**Default value**

None

### partition_random_seed

#### Description

{% include [reusage-cv-rand__desc_intro](../_includes/work_src/reusage/cv-rand__desc_intro.md) %}


{% include [reusage-cv-rand__permutation-is-performed](../_includes/work_src/reusage/cv-rand__permutation-is-performed.md) %}


{% include [reusage-cv-rand__unique-data-splits](../_includes/work_src/reusage/cv-rand__unique-data-splits.md) %}


**Possible types**

{{ python-type--int }}

**Default value**

`0`

### calc_cv_statistics

#### Description

Estimate the quality by using cross-validation with the best of the found parameters. The model is fitted using these parameters.

This option can be enabled if the `search_by_train_test_split` parameter is set to True.


**Possible types**

{{ python-type--bool }}

**Default value**

True

### search_by_train_test_split

#### Description

Split the source dataset into train and test parts. Models are trained on the train part, while parameters are compared by the loss function score on the test dataset.

It is recommended to enable this option for large datasets and disable it for the small ones.

**Possible types**

{{ python-type--bool }}

**Default value**

True

### refit

#### Description

Refit an estimator using the best-found parameters on the whole dataset.

**Possible types**

{{ python-type--bool }}

**Default value**

`True`

### shuffle

#### Description

Shuffle the dataset objects before splitting into folds.

**Possible types**

{{ python-type--bool }}

**Default value**

`True`

### stratified

#### Description

Perform stratified sampling. True for classification and False otherwise.

**Possible types**

{{ python-type--bool }}

**Default value**

{{ python-type--none }}

### train_size

#### Description

The proportion of the dataset to include in the train split.

Possible values are in the range [0;1].

**Possible types**

{{ python-type--float }}

**Default value**

0.8

### verbose

#### Description


{% include [sections-with-methods-desc-python__feature-importances__verbose__short-description__list-intro](../_includes/work_src/reusage/python__feature-importances__verbose__short-description__list-intro.md) %}


- {{ python-type--int }} — The frequency of iterations to print the information to stdout.
- {{ python-type--bool }} — Print the information to stdout on every iteration (if set to <q>True</q>) or disable any logging (if set to <q>False</q>).

**Possible types**


- {{ python-type--bool }}
- {{ python-type--int }}

**Default value**

True

### plot

#### Description


Draw train and evaluation metrics for every set of parameters in Jupyter [Jupyter Notebook](../features/visualization_jupyter-notebook.md).

**Possible types**

{{ python-type--bool }}

**Default value**

False

{% include [python__log-params](../_includes/work_src/reusage-python/python__log-params.md) %}

## {{ dl__return-value }} {#output-format}

Dict with two fields:

- `params` — `dict` of best-found parameters.
- `cv_results` — `dict` or {{ python-type--pandascoreframeDataFrame }} with cross-validation results. Сolumns are: `test-error-mean`, `test-error-std`, `train-error-mean`, `train-error-std`.

## {{ dl--example }} {#example}

```python
from catboost import CatBoostClassifier
import numpy as np

train_data = np.random.randint(1, 100, size=(100, 10))
train_labels = np.random.randint(2, size=(100))

model = CatBoostClassifier(loss_function='Logloss')

grid = {'learning_rate': [0.03, 0.1],
        'depth': [4, 6, 10],
        'l2_leaf_reg': [1, 3, 5, 7, 9]}

grid_search_result = model.grid_search(grid,
                                       X=train_data,
                                       y=train_labels,
                                       plot=True)

```

{% include [reusage-code-examples-graph-plotted-with-jupyter-notebook](../_includes/work_src/reusage-code-examples/graph-plotted-with-jupyter-notebook.md) %}

![](../images/interface__catboostclassifier__grid_search.png)
