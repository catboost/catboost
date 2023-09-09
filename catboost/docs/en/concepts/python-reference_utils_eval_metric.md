# eval_metric

{% include [utils-eval_metric__desc](../_includes/work_src/reusage-python/eval_metric__desc.md) %}


## {{ dl--invoke-format }} {#call-format}

```python
eval_metric(label,
            approx,
            metric,
            weight=None,
            group_id=None,
            subgroup_id=None,
            pairs=None,
            thread_count=-1)
```

## {{ dl--parameters }} {#parameters}

### label

#### Description

A list of target variables (in other words, the label values of the objects).

**Possible types**

- {{ python-type--list }}
- {{ python-type--numpy-ndarray }}
- {{ python-type--pandasDataFrame }}
- {{ python-type--pandasSeries }}

**Default value**

{{ python--required }}

### approx

#### Description

A list of approximate values for all input objects.

**Possible types**

- {{ python-type--list }}
- {{ python-type--numpy-ndarray }}
- {{ python-type--pandasDataFrame }}
- {{ python-type--pandasSeries }}

**Default value**

{{ python--required }}

### metric

#### Description

The evaluation metric to calculate.

{% cut "Supported metrics" %}

{% include [reusage-all-objectives-and-metrics](../_includes/work_src/reusage/all-objectives-and-metrics.md) %}

{% endcut %}

**Possible types**

{{ python-type--string }}

**Default value**

{{ python--required }}

### weight

#### Description

The weights of objects.

**Possible types**

- {{ python-type--list }}
- {{ python-type--numpy-ndarray }}
- {{ python-type--pandasDataFrame }}
- {{ python-type--pandasSeries }}

**Default value**

None

### group_id

#### Description

Group identifiers for all input objects. Supported identifier types are:
- {{ python-type--int }}
- string types ({{ python-type--string }} or {{ python-type__unicode }} for Python 2 and {{ python-type__bytes }} or {{ python-type--string }} for Python 3).

**Possible types**

- {{ python-type--list }}
- {{ python-type--numpy-ndarray }}
- {{ python-type--pandasDataFrame }}
- {{ python-type--pandasSeries }}

**Default value**

None

### subgroup_id

#### Description

Subgroup identifiers for all input objects.

**Possible types**

- {{ python-type--list }}
- {{ python-type--numpy-ndarray }}

**Default value**

None

### pairs

#### Description

The description is different for each group of possible types.

**Possible types**

{% cut "{{ python-type--list }}, {{ python-type--numpy-ndarray }}, {{ python-type--pandasDataFrame }}" %}


The pairs description in the form of a two-dimensional matrix of shape `N` by 2:
- `N` is the number of pairs.
- The first element of the pair is the zero-based index of the winner object from the input dataset for pairwise comparison.
- The second element of the pair is the zero-based index of the loser object from the input dataset for pairwise comparison.

{% include [reusage-learn_pairs__where_is_used](../_includes/work_src/reusage/learn_pairs__where_is_used.md) %}

{% endcut %}

{% cut "{{ python-type--string }}" %}

The path to the input file that contains the [pairs description](../concepts/input-data_pairs-description.md).

{% include [reusage-learn_pairs__where_is_used](../_includes/work_src/reusage/learn_pairs__where_is_used.md) %}

{% endcut %}


**Default value**

None

### thread_count

#### Description

The number of threads to use.

{% include [reusage-thread_count__cpu_cores__optimizes-the-speed-of-execution](../_includes/work_src/reusage/thread_count__cpu_cores__optimizes-the-speed-of-execution.md) %}

**Possible types**

{{ python-type--int }}

**Default value**

{{ fit__thread_count__wrappers }}


## {{ dl--output-format }} {#output-format}

{{ python-type--list }} with metric values.

## {{ dl__usage-examples }} {#usage-examples}

The following is an example of usage with a [regression metric](loss-functions-regression.md):

```python
from catboost.utils import eval_metric

labels = [0.2, -1, 0.4]
predictions = [0.4, 0.1, 0.9]

rmse = eval_metric(labels, predictions, 'RMSE')
```

The following is an example of usage with a [classification metric](loss-functions-classification.md):

```python
from catboost.utils import eval_metric
from math import log

labels = [1, 0, 1]
probabilities = [0.4, 0.1, 0.9]

# In binary classification it is necessary to apply the logit function
# to the probabilities to get approxes.

logit = lambda x: log(x / (1 - x))
approxes = list(map(logit, probabilities))

accuracy = eval_metric(labels, approxes, 'Accuracy')

```

The following is an example of usage with a [ranking metric](loss-functions-ranking.md):

```python
from catboost.utils import eval_metric

# The dataset consists of five objects. The first two belong to one group
# and the other three to another.
group_ids = [1, 1, 2, 2, 2]

labels = [0.9, 0.1, 0.5, 0.4, 0.8]

# In ranking tasks it is not necessary to predict the same labels.
# It is important to predict the right order of objects.
good_predictions = [0.5, 0.4, 0.2, 0.1, 0.3]
bad_predictions = [0.4, 0.5, 0.2, 0.3, 0.1]

good_ndcg = eval_metric(labels, good_predictions, 'NDCG', group_id=group_ids)
bad_ndcg = eval_metric(labels, bad_predictions, 'NDCG', group_id=group_ids)
```
