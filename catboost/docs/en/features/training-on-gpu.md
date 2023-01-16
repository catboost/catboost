# Training on GPU

{{ product }} supports training on GPUs.

Training on GPU is non-deterministic, because the order of floating point summations is non-deterministic in this implementation.

Choose the implementation for more details on the parameters that are required to start training on GPU.

{% if audience == "internal" %}

{% include [gpu-training](../yandex_specific/_includes/gpu-training.md) %}

{% endif %}

## {{ python-package }}

The parameters that enable and customize training on GPU are set in the constructors of the following classes:

- [CatBoost](../concepts/python-reference_catboost.md) ([fit](../concepts/python-reference_catboost_fit.md))
- [CatBoostClassifier](../concepts/python-reference_catboostclassifier.md) ([fit](../concepts/python-reference_catboostclassifier_fit.md))
- [CatBoostRegressor](../concepts/python-reference_catboostregressor.md) ([fit](../concepts/python-reference_catboostregressor_fit.md))

### Parameters

#### task_type

The processing unit type to use for training.

Possible values:
- CPU
- GPU


#### devices

IDs of the GPU devices to use for training (indices are zero-based).

Format

- `<unit ID>` for one device (for example, `3`)
- `<unit ID1>:<unit ID2>:..:<unit IDN>` for multiple devices (for example, `devices='0:1:3'`)
- `<unit ID1>-<unit IDN>` for a range of devices (for example, `devices='0-3'`)


{% note info %}

Other training parameters are also available. Some of them are CPU-specific or GPU-specific. See the [Python package training parameters](../references/training-parameters/index.md) section for more details.

{% endnote %}

For example, use the following code to train a classification model on GPU:
```python
from catboost import CatBoostClassifier

train_data = [[0, 3],
              [4, 1],
              [8, 1],
              [9, 1]]
train_labels = [0, 0, 1, 1]

model = CatBoostClassifier(iterations=1000,
                           task_type="GPU",
                           devices='0:1')
model.fit(train_data,
          train_labels,
          verbose=False)

```

## {{ r-package }}

For the [catboost.train](../concepts/r-reference_catboost-train.md) method:

### {{ features__table__title__r__parameters }}

#### task_type

The processing unit type to use for training.

Possible values:
- CPU
- GPU

#### devices

**{{ features__table__title__r__parameters }}**

IDs of the GPU devices to use for training (indices are zero-based).

Format

- `<unit ID>` for one device (for example, `3`)
- `<unit ID1>:<unit ID2>:..:<unit IDN>` for multiple devices (for example, `devices='0:1:3'`)
- `<unit ID1>-<unit IDN>` for a range of devices (for example, `devices='0-3'`)

For example, use the following code to train a model on GPU:
```r
library(catboost)

dataset = matrix(c(1900,7,
                   1896,1,
                   1896,41),
                 nrow=3,
                 ncol=2,
                 byrow = TRUE)
label_values = c(0,1,1)

fit_params <- list(iterations = 100,
                   loss_function = 'Logloss',
                   task_type = 'GPU')

pool = catboost.load_pool(dataset, label = label_values)

model <- catboost.train(pool, params = fit_params)
```

## {{ title__implementation__cli }}

For the [catboost fit](../references/training-parameters/index.md) command:

### **{{ features__table__title__cli__keys }}**

#### --task-type

The processing unit type to use for training.

Possible values:
- CPU
- GPU

#### --devices

IDs of the GPU devices to use for training (indices are zero-based).

Format

- `<unit ID>` for one device (for example, `3`)
- `<unit ID1>:<unit ID2>:..:<unit IDN>` for multiple devices (for example, `devices='0:1:3'`)
- `<unit ID1>-<unit IDN>` for a range of devices (for example, `devices='0-3'`)

{% include [cli__train-on-gpu-cli__train-on-gpu__p](../_includes/work_src/reusage-code-examples/cli__train-on-gpu__p.md) %}
