# catboost.get_object_importance

```r
catboost.get_object_importance(model,
                               pool,
                               train_pool,
                               top_size = -1,
                               type = '{{ fit__ostr__ostr_type__PerPool }}',
                               update_method = 'SinglePoint',
                               thread_count = -1)
```

## {{ dl--purpose }} {#purpose}

{% include [sections-with-methods-desc-get_object_importance__div](../_includes/work_src/reusage/get_object_importance__div.md) %}

## {{ dl--args }} {#arguments}
### model


#### Description

The model obtained as the result of training.


**Default value**

{{ r--required }}

### pool


#### Description

The input dataset.

{% if audience == "internal" %}

#### For datasets input as files

{% include [files-internal-files-internal__desc__full](../yandex_specific/_includes/reusage-formats/files-only-internal__desc__full.md) %}

{% endif %}

**Default value**

{{ r--required }}

### train_pool


#### Description

The dataset used for training.


**Default value**

{{ r--required }}

### top_size


#### Description

Defines the number of most important objects from the training dataset. The number of returned objects is limited to this number.


**Default value**

{{ fit__ostr__top_size }}

### type


#### Description

The method for calculating the object importances.

Possible values:
- {{ fit__ostr__ostr_type__PerPool }} — The average of scores of objects from the training dataset for every object from the input dataset.
- {{ fit__ostr__ostr_type__PerObject }} — The scores of each object from the training dataset for each object from the input dataset.


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
```no-highlight
TopKLeaves:top=3
```


**Default value**

{{ ostr__update-method__default }}

### thread_count


#### Description

{% include [reusage-thread-count-short-desc](../_includes/work_src/reusage/thread-count-short-desc.md) %}


{% include [reusage-thread_count__cpu_cores__optimizes-the-speed-of-execution](../_includes/work_src/reusage/thread_count__cpu_cores__optimizes-the-speed-of-execution.md) %}



**Default value**

{{ fit__thread_count__wrappers }}

## {{ dl--example }} {#example}

{% include [ostr__r-object-strength__r__p](../_includes/work_src/reusage-code-examples/object-strength__r__p.md) %}


