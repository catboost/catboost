# Sum models

## {{ dl--purpose }}

{% include [sum_limits-python__sum-limits__desc](../_includes/work_src/reusage-python/python__sum-limits__desc.md) %}


## {{ dl__cli__execution-format }} {#execution-format}

```
catboost model-sum -m [<model with 1.0 weight>] [--model-with-weight <path_to_model>=<weight>] -o <result model path>
```

## {{ common-text__title__reference__parameters }} {#options}

### -m

#### Description

The path to the model that should be blended with the others. The weight of the leafs of this model is set to 1.

This parameter can be specified several times to set the required number of input models.

**{{ cli__params-table__title__default }}**

At least one of `--m`, `-model-with-weight` parameters should be set at least once.

### --model-with-weight

#### Description

The path to the model that should be blended with the others. Use this parameter to set the individual weight for the leaf values of this model. Use the equal sign as the separator.

This parameter can be specified several times to set the required number of input models.


**{{ cli__params-table__title__default }}**

At least one of `--m`, `-model-with-weight` parameters should be set at least once.

### -o

#### Description

The path to the output model obtained as the result of blending the input ones.

**{{ cli__params-table__title__default }}**

 {{ python--required }}

### --ctr-merge-policy

#### Description

The counters merging policy. Possible values:
- {{ ECtrTableMergePolicy__FailIfCtrIntersects }} — Ensure that the models have zero intersecting counters.
- {{ ECtrTableMergePolicy__LeaveMostDiversifiedTable }} — Use the most diversified counters by the count of unique hash values.
- {{ ECtrTableMergePolicy__IntersectingCountersAverage }} — Use the average ctr counter values in the intersecting bins.

**{{ cli__params-table__title__default }}**

 {{ ECtrTableMergePolicy__default }}
