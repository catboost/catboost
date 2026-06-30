### pool

#### Description

The data for calculating object importances.

**Possible types**

{{ python-type--pool }}

**Default value**

{{ python--required }}


### train_pool

#### Description

The dataset used for training.

**Possible types**

{{ python-type--pool }}

**Default value**

{{ python--required }}


### top_size

#### Description

Defines the number of most important objects from the training dataset. The number of returned objects is limited to this number.


**Possible types**

{{ python-type--int }}

**Default value**

{{ fit__ostr__top_size }}


### type

#### Description

The method for calculating the object importances.

Possible values:
- {{ fit__ostr__ostr_type__PerPool }} — The average of scores of objects from the training dataset for every object from the input dataset.
- {{ fit__ostr__ostr_type__PerObject }} — The scores of each object from the training dataset for each object from the input dataset.

**Possible types**

{{ python-type--string }}

**Default value**

{{ fit__ostr__ostr_type }}


### update_method

#### Description
The algorithm accuracy method.

Possible values:
- {{ ostr__update-method__SinglePoint }} — The fastest and least accurate method.
- {{ ostr__update-method__TopKLeaves }} — Specify the number of leaves. The higher the value, the more accurate and the slower the calculation.
- {{ ostr__update-method__AllPoints }} — The slowest and most accurate method.

Supported parameters:
- `top` — Defines the number of leaves to use for the {{ ostr__update-method__TopKLeaves }} update method. See the [Finding Influential Training Samples for Gradient Boosted Decision Trees](https://arxiv.org/abs/1802.06640) for more details.

For example, the following value sets the method to {{ ostr__update-method__TopKLeaves }} and limits the number of leaves to 3:
```
TopKLeaves:top=3
```
**Possible types**

{{ python-type--string }}

**Default value**

{{ ostr__update-method__default }}


### importance_values_sign

#### Description

Defines the type of effect that the objects from the training dataset must have on the optimized metric value for objects from the input dataset. Only the appropriate objects are output.
Possible values:
- {{ fit__ostr__importance_values_sign__Positive }}
- {{ fit__ostr__importance_values_sign__Negative }}
- {{ fit__ostr__importance_values_sign__All }}

**Possible types**

{{ python-type--string }}

**Default value**

{{ fit__ostr__importance_values_sign }}


### thread_count

#### Description

{% include [reusage-thread-count-short-desc](thread-count-short-desc.md) %}


{% include [reusage-thread_count__cpu_cores__optimizes-the-speed-of-execution](thread_count__cpu_cores__optimizes-the-speed-of-execution.md) %}

**Possible types**

{{ python-type--int }}

**Default value**

{{ fit__thread_count__wrappers }}

{% include [python__log-params](../../../_includes/work_src/reusage-python/python__log-params.md) %}
