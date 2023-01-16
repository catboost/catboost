### -f

#### Description

The path to the dataset to cross-validate.

**{{ cli__params-table__title__default }}**

 Required parameter (the path must be specified).

### --cv

#### Description

Enable the cross-validation mode and specify the launching parameters.

Format:

```
<cv_type>:<fold_index>;<fold_count>
```

The following cross-validation types (`cv_type`) are supported:

{% list tabs %}

- {{ cv__type__Classical }}

    Format: `{{ cv__type__Classical }}:<fold_index>;<fold_count>`

    - `fold_index` is the index of the fold to exclude from the learning data and use for evaluation (indexing starts from zero).

    - {% include [common-phrases-cv__k-param__desc](cv__k-param__desc.md) %}

    All folds, except the one indexed `n`, are used as the learning dataset. The fold indexed `n` is used as the validation dataset.

    {% include [common-phrases-cv__n_less_than_K](cv__n_less_than_K.md) %}

    {% include [common-phrases-cv__random_shuffling](cv__random_shuffling.md) %}

- {{ cv__type__Inverted }}

    Format: `{{ cv__type__Inverted }}:<fold_index>;<fold_count>`
    - `fold_index` is the index of the fold to use for learning (indexing starts from zero).
    - {% include [common-phrases-cv__k-param__desc](cv__k-param__desc.md) %}

    The fold indexed `fold_index` is used as the learning dataset. All other folds are used as the validation dataset.

    {% include [common-phrases-cv__n_less_than_K](cv__n_less_than_K.md) %}

    {% include [common-phrases-cv__random_shuffling](cv__random_shuffling.md) %}

{% endlist %}

{% cut "Example" %}

Split the input dataset into 5 folds, use the one indexed 0 for validation and all others for training:

```
--cv Classical:0;5
```

{% endcut %}

**{{ cli__params-table__title__default }}**

 Required parameter for cross-validation

### --cv-rand

#### Description

Use this as the seed value for random permutation of the data.

The permutation is performed before splitting the data for cross-validation.

Each seed generates unique data splits.

It must be used with the `--cv` parameter type set to {{ cv__type__Classical }} or {{ cv__type__Inverted }}.

**{{ cli__params-table__title__default }}**

 {{ fit--cv-rand }}

### --cv-no-shuffle

#### Description

Do not shuffle the dataset before cross-validation.

**{{ cli__params-table__title__default }}**

 {{ fit--cv-no-shuffle }}

### other parameters

#### Description

Any combination of the [training parameters](../../../references/training-parameters/index.md).

**{{ cli__params-table__title__default }}**

 See the full list of default values in the [Train a model](../../../references/training-parameters/index.md) section.
