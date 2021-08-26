# catboost.train

```no-highlight
catboost.train(learn_pool,
               test_pool = NULL,
               params = list())
```

## {{ dl--purpose }} {#purpose}

{% include [reusage-r-train__purpose](../_includes/work_src/reusage-r/train__purpose.md) %}


{% note info %}

{% include [installation-nvidia-driver-reqs](../_includes/work_src/reusage-code-examples/nvidia-driver-reqs.md) %}

{% endnote %}


## {{ dl--args }} {#arguments}

### learn_pool

#### Description

The dataset used for training the model.

**Default value**

 {{ r--required }}

### test_pool

#### Description

The dataset used for testing the quality of the model.

**Default value**

 NULL (not used)

### params

#### Description

The list of parameters to start training with.

If omitted, default values are used (refer to the  [{{ dl--parameters }}](#parameters) section).

If set, the passed list of parameters overrides the default values.

**Default value**

 {{ r--required }}

## {{ dl--parameters }} {#parameters}

{% include [parameters-params-list](../_includes/work_src/reusage/params-list.md) %}

## {{ dl--example }} {#example}

{% include [train-training-parameters-and-fit](../_includes/work_src/reusage-code-examples/training-parameters-and-fit.md) %}


{% include [train-training-parameters-and-fit-on-gpu](../_includes/work_src/reusage-code-examples/training-parameters-and-fit-on-gpu.md) %}


{% include [train-dataset-with-cat-features](../_includes/work_src/reusage-code-examples/dataset-with-cat-features.md) %}


{% include [train-dataset-with-cat-features-with-gpu](../_includes/work_src/reusage-code-examples/dataset-with-cat-features-with-gpu.md) %}
