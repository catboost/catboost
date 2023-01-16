# catboost.staged_predict

```
catboost.staged_predict(model,
                        pool,
                        verbose=FALSE,
                        prediction_type=None,
                        ntree_start=0,
                        ntree_end=0,
                        eval_period=1,
                        thread_count=-1)
```

## {{ dl--purpose }} {#purpose}

{% include [reusage-r-staged_predict__purpose](../_includes/work_src/reusage-r/staged_predict__purpose.md) %}


{% note info %}

{% include [predict__note__text__for__all_packages-r__note__predict_note_for_packages_must-contain-all-features__r](../_includes/work_src/reusage-common-phrases/r__note__predict_note_for_packages_must-contain-all-features__r.md) %}

{% endnote %}


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

### verbose


#### Description

Verbose output to stdout.

**Default value**

{{ fit--verbose-r }}

### prediction_type


#### Description


The required prediction type.

Supported prediction types:
- {{ prediction-type--Probability }}
- {{ prediction-type--Class }}
- {{ prediction-type--RawFormulaVal }}
- {{ prediction-type--Exponent }}
- {{ prediction-type--LogProbability }}


**Default value**

None ({{ prediction-type--Exponent }} for {{ error-function--Poisson }} and {{ error-function__Tweedie }}, {{ prediction-type--RawFormulaVal }} for all other loss functions)

### ntree_start


#### Description


To reduce the number of trees to use when the model is applied or the metrics are calculated, setthe range of the tree indices to`[ntree_start; ntree_end)` and the the step of the trees to use to`eval_period`.

{% include [eval-start-end-ntree_start__short-param-desc](../_includes/work_src/reusage-common-phrases/ntree_start__short-param-desc.md) %}



**Default value**

{{ fit--ntree_start }}

### ntree_end


#### Description


To reduce the number of trees to use when the model is applied or the metrics are calculated, setthe range of the tree indices to`[ntree_start; ntree_end)` and the the step of the trees to use to`eval_period`.

{% include [eval-start-end-ntree_end__short-param-desc](../_includes/work_src/reusage-common-phrases/ntree_end__short-param-desc.md) %}



**Default value**

{{ fit--ntree_end }}

### eval_period


#### Description


To reduce the number of trees to use when the model is applied or the metrics are calculated, setthe range of the tree indices to`[ntree_start; ntree_end)` and the the step of the trees to use to`eval_period`.

{% include [eval-start-end-python_r__eval__period__desc](../_includes/work_src/reusage-common-phrases/python_r__eval__period__desc.md) %}


**Default value**

{{ fit--staged-predict-eval-period }}

### thread_count


#### Description


{% include [reusage-thread-count-short-desc](../_includes/work_src/reusage/thread-count-short-desc.md) %}


{% include [reusage-thread_count__cpu_cores__optimizes-the-speed-of-execution](../_includes/work_src/reusage/thread_count__cpu_cores__optimizes-the-speed-of-execution.md) %}



**Default value**

{{ fit__thread_count__wrappers }}

