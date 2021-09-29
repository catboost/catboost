# Cross-validation

{{ product }} allows to perform cross-validation on the given dataset.

{% include [feature-importance-choose-the-required-implementation-for-more-details](../_includes/work_src/reusage-common-phrases/choose-the-required-implementation-for-more-details.md) %}


## {{ python-package }}

### {{ features__table__title__python__class }}

[cv](../concepts/python-reference_cv.md)

**{{ features__table__title__python__class-applicability }}**

{% include [cv-cv__purpose](../_includes/work_src/reusage-python/cv__purpose.md) %}

## {{ title__implementation__cli }}

For the [catboost fit](../concepts/cli-reference_cross-validation.md) command:

**{{ features__table__title__cli__purpose }}**

{% include [reusage-cli__cross-validation__purpose__div](../_includes/work_src/reusage/cli__cross-validation__purpose__div.md) %}

### {{ features__table__title__cli__keys }}

#### --cv

**{{ features__table__title__cli__key-description }}**

Enable the cross-validation mode and specify the launching parameters.

Format:
```no-highlight
<cv_type>:<fold_index>;<fold_count>
```

The following cross-validation types (`cv_type`) are supported:

##### {{ cv__type__Classical }}

Format: `{{ cv__type__Classical }}<fold_index>;<fold_count>`

- `fold_index` is the index of the fold to exclude from the learning data and use for evaluation (indexing starts from zero).

- {% include [common-phrases-cv__k-param__desc](../_includes/work_src/reusage/cv__k-param__desc.md) %}


All folds, except the one indexed `n`, are used as the learning dataset. The fold indexed `n` is used as the validation dataset.

{% include [common-phrases-cv__n_less_than_K](../_includes/work_src/reusage/cv__n_less_than_K.md) %}


{% include [common-phrases-cv__random_shuffling](../_includes/work_src/reusage/cv__random_shuffling.md) %}


##### {{ cv__type__Inverted }}

Format: `{{ cv__type__Inverted }}<fold_index>;<fold_count>`
- `fold_index` is the index of the fold to use for learning (indexing starts from zero).
- {% include [common-phrases-cv__k-param__desc](../_includes/work_src/reusage/cv__k-param__desc.md) %}

The fold indexed `fold_index` is used as the learning dataset. All other folds are used as the validation dataset.

{% include [common-phrases-cv__n_less_than_K](../_includes/work_src/reusage/cv__n_less_than_K.md) %}


{% include [common-phrases-cv__random_shuffling](../_includes/work_src/reusage/cv__random_shuffling.md) %}

{% cut "Example" %}

Split the input dataset into 5 folds, use the one indexed 0 for validation and all others for training:

```no-highlight
--cv Classical:0;5
```

{% endcut %}

#### --cv-rand

**{{ features__table__title__cli__purpose }}**

Use this as the seed value for random permutation of the data.

The permutation is performed before splitting the data for cross-validation.

Each seed generates unique data splits.

It must be used with the `--cv` parameter type set to {{ cv__type__Classical }} or {{ cv__type__Inverted }}.

