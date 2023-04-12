# CTR settings

## simple_ctr {#simple_ctr}

#### Description


{% include [reusage-cli__simple-ctr__intro](../../_includes/work_src/reusage/cli__simple-ctr__intro.md) %}


Format:

```
['CtrType[:{{ ctr-types__TargetBorderCount }}=BorderCount][:TargetBorderType=BorderType][:CtrBorderCount=Count][:CtrBorderType=Type][:Prior=num_1/denum_1]..[:Prior=num_N/denum_N]',
 'CtrType[:{{ ctr-types__TargetBorderCount }}=BorderCount][:TargetBorderType=BorderType][:CtrBorderCount=Count][:CtrBorderType=Type][:Prior=num_1/denum_1]..[:Prior=num_N/denum_N]',
  ...]
```

{% include [reusage-cli__simple-ctr__components](../../_includes/work_src/reusage/cli__simple-ctr__components.md) %}


{% include [reusage-cli__simple-ctr__examples__p](../../_includes/work_src/reusage/cli__simple-ctr__examples__p.md) %}


**Type**

{{ python-type--string }}

**Supported processing units**

{{ cpu-gpu }}


## combinations_ctr {#combinations_ctr}

#### Description


{% include [reusage-cli__combination-ctr__intro](../../_includes/work_src/reusage/cli__combination-ctr__intro.md) %}


```
['CtrType[:{{ ctr-types__TargetBorderCount }}=BorderCount][:TargetBorderType=BorderType][:CtrBorderCount=Count][:CtrBorderType=Type][:Prior=num_1/denum_1]..[:Prior=num_N/denum_N]',
 'CtrType[:{{ ctr-types__TargetBorderCount }}=BorderCount][:TargetBorderType=BorderType][:CtrBorderCount=Count][:CtrBorderType=Type][:Prior=num_1/denum_1]..[:Prior=num_N/denum_N]',
  ...]
```

{% include [reusage-cli__combination-ctr__components](../../_includes/work_src/reusage/cli__combination-ctr__components.md) %}

**Type**

{{ python-type--string }}

**Supported processing units**

{{ cpu-gpu }}



## per_feature_ctr {#per_feature_ctr}

#### Description

{% include [reusage-cli__per-feature-ctr__intro](../../_includes/work_src/reusage/cli__per-feature-ctr__intro.md) %}


```
['FeatureId:CtrType:[:{{ ctr-types__TargetBorderCount }}=BorderCount][:TargetBorderType=BorderType][:CtrBorderCount=Count][:CtrBorderType=Type][:Prior=num_1/denum_1]..[:Prior=num_N/denum_N]',
 'FeatureId:CtrType:[:{{ ctr-types__TargetBorderCount }}=BorderCount][:TargetBorderType=BorderType][:CtrBorderCount=Count][:CtrBorderType=Type][:Prior=num_1/denum_1]..[:Prior=num_N/denum_N]',
  ...]
```

{% include [reusage-cli__per-feature-ctr__components](../../_includes/work_src/reusage/cli__per-feature-ctr__components.md) %}

**Type**

{{ python-type--string }}

**Supported processing units**

{{ cpu-gpu }}


## ctr_target_border_count {#ctr_target_border_count}

#### Description


{% include [reusage-cli__ctr-target-border-count__short-desc](../../_includes/work_src/reusage/cli__ctr-target-border-count__short-desc.md) %}


The value of the `{{ ctr-types__TargetBorderCount }}` component overrides this parameter if it is specified for one of the following parameters:

- `simple_ctr`
- `combinations_ctr`
- `per_feature_ctr`

**Type**

{{ python-type--int }}

**Default value**

{{ parameters__ctr-target-border-count__default }}

**Supported processing units**

{{ cpu-gpu }}


## counter_calc_method {#counter_calc_method}

#### Description

The method for calculating the Counter CTR type.

Possible values:
- {{ counter-calculation-method--static }} — Objects from the validation dataset are not considered at all
- {{ counter-calculation-method--full }} — All objects from both learn and validation datasets are considered

**Type**

{{ python-type--string }}

**Default value**

None ({{ fit--counter-calc-method }} is used)

**Supported processing units**

{{ cpu-gpu }}

## max_ctr_complexity {#max_ctr_complexity}

#### Description


The maximum number of features that can be combined.

Each resulting combination consists of one or more categorical features and can optionally contain binary features in the following form: <q>numeric feature > value</q>.

**Type**

{{ python-type--int }}

**Default value**

{% include [reusage-default-values-max_xtr_complexity](../../_includes/work_src/reusage-default-values/max_xtr_complexity.md) %}

**Supported processing units**

{{ cpu-gpu }}



## ctr_leaf_count_limit {#ctr_leaf_count_limit}

#### Description


The maximum number of leaves with categorical features. If the quantity exceeds the specified value a part of leaves is discarded.

The leaves to be discarded are selected as follows:

1. The leaves are sorted by the frequency of the values.
1. The top `N` leaves are selected, where N is the value specified in the parameter.
1. All leaves starting from `N+1` are discarded.

This option reduces the resulting model size and the amount of memory required for training. Note that the resulting quality of the model can be affected.

**Type**

{{ python-type--int }}

**Default value**

None

{{ fit--ctr-leaf-count-limit }}

**Supported processing units**

{{ calcer_type__cpu }}

## store_all_simple_ctr {#store_all_simple_ctr}

#### Description


Ignore categorical features, which are not used in feature combinations, when choosing candidates for exclusion.

There is no point in using this parameter without the `--ctr-leaf-count-limit` for the Command-line version parameter.

**Type**

{{ python-type--bool }}

**Default value**

None (set to False)

{{ fit--store-all-simple-ctr }}

**Supported processing units**

{{ calcer_type__cpu }}

## final_ctr_computation_mode {#final_ctr_computation_mode}

#### Description


Final CTR computation mode.

Possible values:
- {{ cli__fit__final-ctr-computation-mode__possible-values__Default }} — Compute final CTRs for learn and validation datasets.
- {{ cli__fit__final-ctr-computation-mode__possible-values__Skip }} — Do not compute final CTRs for learn and validation datasets. In this case, the resulting model can not be applied. This mode decreases the size of the resulting model. It can be useful for research purposes when only the metric values have to be calculated.

**Type**

{{ python-type--string }}

**Default value**

{{ cli__fit__final-ctr-computation-mode__default }}

**Supported processing units**

{{ cpu-gpu }}


