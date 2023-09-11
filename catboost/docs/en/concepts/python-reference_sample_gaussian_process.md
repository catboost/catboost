# sample_gaussian_process

## {{ dl--purpose }} {#purpose}

Implementation of Gaussian process sampling (Kernel Gradient Boosting/Algorithm 4) from ["Gradient Boosting Performs Gaussian Process Inference"](https://arxiv.org/abs/2206.05608) paper.

Produces samples from posterior GP with prior assumption $f \sim \mathcal{GP}(0, \sigma^2 \mathcal{K} + \delta^2 I)$

## {{ dl--invoke-format }} {#call-format}

```python
sample_gaussian_process(X,
                        y,
                        eval_set=None,
                        cat_features=None,
                        text_features=None,
                        embedding_features=None,
                        random_seed=None,
                        samples=10,
                        posterior_iterations=900,
                        prior_iterations=100,
                        learning_rate=0.1,
                        depth=6,
                        sigma=0.1,
                        delta=0,
                        random_strength=0.1,
                        random_score_type='Gumbel',
                        eps=1e-4,
                        verbose=False)
```

## {{ dl--parameters }} {#parameters}

### X

#### Description

Training data with features.
Must be non-empty (contain > 0 objects)

**Possible types**

{% cut "{{ python-type--list }}, {{ python-type--numpy-ndarray }}, {{ python-type--pandasDataFrame }}, {{ python-type--pandasSeries }}" %}

Two-dimensional feature matrix.

{% endcut %}


{% cut "{{ python_type__pandas-SparseDataFrame }}, {{ python_type__scipy-sparse-spmatrix }} (all subclasses except dia_matrix)" %}

Two-dimensional sparse feature matrix.

{% endcut %}


{% cut "{{ python-type__FeaturesData }}" %}

Special class for features data. See [FeaturesData](python-features-data__desc.md).

{% endcut %}


**Default value**

{{ python--required }}


### y

#### Description

Training data labels (numerical). 1-dimensional array like.

**Possible types**

- {{ python-type--list }}
- {{ python-type--numpy-ndarray }}
- {{ python-type--pandasSeries }}

**Default value**

{{ python--required }}


### eval_set

#### Description

The validation dataset or datasets used for the following processes in the posterior fitting:
- [overfitting detector](../concepts/overfitting-detector.md)
- monitoring metrics' changes

**Possible types**

- {{ python-type--pool }}
- {{ python-type--list }} of {{ python-type--pool }}
- {{ python-type--tuple }} (X, y)
- {{ python-type--list }} of {{ python-type--tuple }}s (X, y)
- {{ python-type--string }} (path to the dataset file)
- {{ python-type--list }} of {{ python-type--string }}s (paths to dataset files)

**Default value**

None


### cat_features

#### Description

A one-dimensional array of categorical features columns indices.

**Possible types**

- {{ python-type--list }}
- {{ python-type--numpy-ndarray }}

{% include [default-value-none](../_includes/concepts/default-value-none.md) %}


### text_features

#### Description

A one-dimensional array of text features columns indices.

**Possible types**

- {{ python-type--list }}
- {{ python-type--numpy-ndarray }}

{% include [default-value-none](../_includes/concepts/default-value-none.md) %}


### embedding_features

#### Description

A one-dimensional array of embedding features columns indices.

**Possible types**

- {{ python-type--list }}
- {{ python-type--numpy-ndarray }}

{% include [default-value-none](../_includes/concepts/default-value-none.md) %}


### random_seed

#### Description

The random seed used for training.

**Possible types**

- {{ python-type--int }}

**Default value**

{% include [default-value-none](../_includes/concepts/default-value-none.md) %}


### samples

#### Description

Number of Monte-Carlo samples from GP posterior. Controls how many models this function will return.

Possible range is [1, +inf)

**Possible types**

- {{ python-type--int }}

**Default value**

10


### posterior_iterations

#### Description

Max count of trees for posterior sampling step.

Possible range is \[1, +inf\)

**Possible types**

- {{ python-type--int }}

**Default value**

900


### prior_iterations

#### Description

Max count of trees for prior sampling step.

Possible range is \[1, +inf\)

**Possible types**

- {{ python-type--int }}

**Default value**

100


### learning_rate

#### Description

Step size shrinkage used in update to prevent overfitting.

Possible range is \(0, 1\]

**Possible types**

- {{ python-type--float }}

**Default value**

0.1


### depth

#### Description

Depth of the trees in the models.

Possible range is \[1, 16\]

**Possible types**

- {{ python-type--int }}

**Default value**

6


### sigma

#### Description

Scale of GP kernel (lower values lead to lower posterior variance).

Possible range is \(0, +inf\)

**Possible types**

- {{ python-type--float }}

**Default value**

0.1


### delta

#### Description

Scale of homogenious noise of GP kernel (adjust if target is noisy)

Possible range is \[0, +inf\)

**Possible types**

- {{ python-type--float }}

**Default value**

0.0


### random_strength

#### Description

Corresponds to parameter `beta` in the paper. Higher values lead to faster convergence to GP posterior.

Possible range is \(0, +inf\)

**Possible types**

- {{ python-type--float }}

**Default value**

0.1


### random_score_type

#### Description

Type of random noise added to scores.
Possible values:

- `Gumbel` - Gumbel-distributed (as in paper)
- `NormalWithModelSizeDecrease` - Normally-distributed with deviation decreasing with model iteration count (default in CatBoost)

**Possible types**

- {{ python-type--string }}

**Default value**

- `Gumbel`


### eps

#### Description

Technical parameter that controls the precision of prior estimation.

Possible range is \(0, 1\]

**Possible types**

- {{ python-type--float }}

**Default value**

1.e-4


### verbose

#### Description

Verbosity of posterior model training output
If `verbose` is `bool`, then if set to `True`, `logging_level` is set to `Verbose`,
if set to `False`, `logging_level` is set to `Silent`.
If `verbose` is `int`, it determines the frequency of writing metrics to output and
`logging_level` is set to `Verbose`.

**Possible types**

- {{ python-type--bool }}
- {{ python-type--int }}

**Default value**

`False`


## {{ dl__return-value }} {#output-format}

List of trained [CatBoostRegressor](python-reference_catboostregressor.md) models (size = [samples parameter](#samples) value).
