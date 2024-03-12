# randomized_search

{% include [randomized_search-python__randomized_search__desc__full-div](../_includes/work_src/reusage-python/python__randomized_search__desc__full-div.md) %}


## {{ dl--invoke-format }} {#method-call-format}

```python
randomized_search(param_distributions,
                  X,
                  y=None,
                  cv=3,
                  n_iter=10,
                  partition_random_seed=0,
                  calc_cv_statistics=True,
                  search_by_train_test_split=True,
                  refit=True,
                  shuffle=True,
                  train_size=0.8,
                  verbose=True,
                  log_cout=sys.stdout,
                  log_cerr=sys.stderr)
```

## {{ dl--parameters }} {#parameters}

### param_distributions

#### Description

Dictionary with parameters names (string) as keys and distributions or lists of parameter settings to try. Distributions must provide a `rvs` method for sampling (such as those from `scipy.stats.distributions`).

If a list is given, it is sampled uniformly.

**Possible types:**

dict

**Default value**

Required parameter

### X

#### Description

The description is different for each group of possible types.

**Possible types**

{% cut "{{ python-type--pool }}" %}

The input training dataset.

{% note info %}

If a nontrivial value of the cat_features parameter is specified in the constructor of this class, CatBoost checks the equivalence of categorical features indices specification from the constructor parameters and in this Pool class.

{% endnote %}

{% endcut %}

{% cut "{{ python-type--numpy-ndarray }}, {{ python-type--pandasDataFrame }}" %}

The input training dataset in the form of a two-dimensional feature matrix.

{% endcut %}

{% cut "{{ python_type__pandas-SparseDataFrame }}, scipy.sparse.spmatrix (all subclasses except dia_matrix)" %}

The input training dataset in the form of a two-dimensional sparse feature matrix.

{% endcut %}

**Default value**

Required parameter

### y

#### Description

{% include [methods-param-desc-label--short-desc-training](../_includes/work_src/reusage/label--short-desc-training.md) %}

{% include [methods-param-desc-label--detailed-desc-ranking](../_includes/work_src/reusage/label--detailed-desc-ranking.md) %}

{% note info %}

Do not use this parameter if the input training dataset (specified in the X parameter) type is catboost.Pool.

{% endnote %}

{% include [methods-param-desc-label--possible-types-1d-default-supported-processing-units](../_includes/work_src/reusage/label--possible-types-1d-default-supported-processing-units.md) %}

### cv

#### Description

The cross-validation splitting strategy.
The interpretation of this parameter depends on the input data type:
- None — Use the default three-fold cross-validation.
- int — The number of folds in a (Stratified)KFold
- object — One of the scikit-learn Splitter Classes with the split method.
- An iterable yielding train and test splits as arrays of indices.

**Possible types**

- int
- scikit-learn splitter object
- cross-validation generator
- iterable

**Default value**

None

### n_iter

#### Description

The number of parameter settings that are sampled. This parameter trades off runtime vs quality of the solution.

**Possible types**

int

**Default value**

0

### partition_random_seed

#### Description

Use this as the seed value for random permutation of the data.
The permutation is performed before splitting the data for cross-validation.
Each seed generates unique data splits.

**Possible types**

int

**Default value**

0

### calc_cv_statistics

#### Description

Estimate the quality by using cross-validation with the best of the found parameters. The model is fitted using these parameters.
This option can be enabled if the `search_by_train_test_split` parameter is set to True.

**Possible types**

bool

**Default value**

True

### search_by_train_test_split

#### Description

Split the source dataset into train and test parts. Models are trained on the train part, while parameters are compared by the loss function score on the test dataset.

It is recommended to enable this option for large datasets and disable it for the small ones.

**Possible types**

bool

**Default value**

True

### refit

#### Description

Refit an estimator using the best-found parameters on the whole dataset.

**Possible types**

bool

**Default value**

True

### shuffle

#### Description

Shuffle the dataset objects before splitting into folds.

**Possible types**

bool

**Default value**

True

### train_size

#### Description

The proportion of the dataset to include in the train split.

Possible values are in the range [0;1].

**Possible types**

float

**Default value**

0.8

### verbose

#### Description

The purpose of this parameter depends on the type of the given value:

- int — The frequency of iterations to print the information to stdout.
- bool — Print the information to stdout on every iteration (if set to “True”) or disable any logging (if set to “False”).

**Possible types**

- bool
- int

**Default value**

True

### plot

#### Description

Draw train and evaluation metrics for every set of parameters in Jupyter [Jupyter Notebook](../features/visualization_jupyter-notebook.md).

**Possible types**

bool

**Default value**
False

{% include [python__log-params](../_includes/work_src/reusage-python/python__log-params.md) %}

## {{ dl__return-value }} {#output-format}

Dict with two fields:

- `params` — `dict` of best-found parameters.
- `cv_results` — `dict` or {{ python-type--pandascoreframeDataFrame }} with cross-validation results. Сolumns are: `test-error-mean`, `test-error-std`, `train-error-mean`, `train-error-std`.

## {{ dl--example }} {#example}

{% include [random_search-randomized-search](../_includes/work_src/reusage-code-examples/randomized-search.md) %}
