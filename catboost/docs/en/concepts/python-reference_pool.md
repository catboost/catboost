# Pool

```python
class Pool(data,
           label=None,
           cat_features=None,
           text_features=None,
           embedding_features=None,
           column_description=None,
           pairs=None,
           delimiter='\t',
           has_header=False,
           weight=None,
           group_id=None,
           group_weight=None,
           subgroup_id=None,
           pairs_weight=None,
           baseline=None,
           timestamp=None,
           feature_names=None,
           thread_count=-1,
           log_cout=sys.stdout,
           log_cerr=sys.stderr)
```

## {{ dl--purpose }} {#purpose}

Dataset processing.

The fastest way to pass the features data to the Pool constructor (and other [CatBoost](python-reference_catboost.md), [CatBoostClassifier](python-reference_catboostclassifier.md), [CatBoostRegressor](python-reference_catboostregressor.md) methods that accept it) if most (or all) of your features are numerical is to pass it using FeaturesData class. Another way to get similar performance with datasets that contain numerical features only is to pass features data as numpy.ndarray with numpy.float32 dtype.

## {{ dl--parameters }} {#parameters}

### data

#### Description

The description is different for each group of possible types.

**Possible types**

{% cut "{{ python-type--list }}, {{ python-type--numpy-ndarray }}, {{ python-type--pandasDataFrame }}, {{ python-type--pandasSeries }}" %}

Dataset in the form of a two-dimensional feature matrix.

{% endcut %}


{% cut "{{ python_type__pandas-SparseDataFrame }}, {{ python_type__scipy-sparse-spmatrix }} (all subclasses except dia_matrix)" %}


{% include [libsvm-libsvm__desc](../_includes/work_src/reusage-formats/libsvm__desc.md) %}

{% endcut %}


{% cut "catboost.FeaturesData" %}

Dataset in the form of {{ python-type__FeaturesData }}. The fastest way to create a Pool from Python objects.

{% include [files-internal-files-internal__desc__full](../_includes/work_src/reusage-formats/files-internal__desc__full.md) %}

{% endcut %}

{% cut "{{ python-type--string }}" %}


The path to the input file{% if audience == "internal" %} or table{% endif %} that contains the dataset description.

{% endcut %}


**Default value**

{{ python--required }}


### label

#### Description

The target variables (in other words, the objects' label values).

{% include [methods-param-desc-label--detailed-desc-generic](../_includes/work_src/reusage/label--detailed-desc-generic.md) %}

{% note info %}

If `data` parameter points to a file, label data is loaded from it as well. This parameter must be `None` in this case.

{% endnote %}

**Possible types**

- {{ python-type--list }}
- {{ python-type--numpy-ndarray }}
- {{ python-type--pandasSeries }}
- {{ python-type--pandasDataFrame }}

**Default value**

None

### cat_features

#### Description

A one-dimensional array of categorical columns indices (specified as integers) or names (specified as strings).

Use only if the `data` parameter is a two-dimensional feature matrix (has one of the following types: {{ python-type--list }}, {{ python-type__np_ndarray }}, {{ python-type--pandasDataFrame }}, {{ python-type--pandasSeries }}).

If any elements in this array are specified as names instead of indices, names for all columns must be provided. To do this, either use the `feature_names` parameter of this constructor to explicitly specify them or pass a {{ python-type--pandasDataFrame }} with column names specified in the `data` parameter.

**Possible types**

- {{ python-type--list }}
- {{ python-type--numpy-ndarray }}

**Default value**

None (it is assumed that all columns are the values of numerical features)

### text_features

#### Description

A one-dimensional array of text columns indices (specified as integers) or names (specified as strings).

{% include [reusage-python__cat_features__description__non-catfeatures-text](../_includes/work_src/reusage/python__cat_features__description__non-catfeatures-text.md) %}

**Possible types**

- {{ python-type--list }}
- {{ python-type--numpy-ndarray }}

**Default value**

{% include [default-value-none](../_includes/concepts/default-value-none.md) %}

### embedding_features

#### Description

A one-dimensional array of embedding columns indices (specified as integers) or names (specified as strings).

{% include [reusage-python__cat_features__description__non-catfeatures-text](../_includes/work_src/reusage/python__cat_features__description__non-catfeatures-text.md) %}

**Possible types**

- {{ python-type--list }}
- {{ python-type--numpy-ndarray }}

**Default value**

{% include [default-value-none](../_includes/concepts/default-value-none.md) %}

### column_description

#### Description

The path to the input file {% if audience == "internal" %}or table{% endif %} that contains the [columns description](../concepts/input-data_column-descfile.md).

{% if audience == "internal" %}

{% include [internal__cd-internal-cd-desc](../yandex_specific/_includes/reusage-formats/internal-cd-desc.md) %}

{% endif %}

**Possible types**

{{ python-type--string }}

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

### delimiter

#### Description

The delimiter character used to separate the data in the dataset description input file.

Only single char delimiters are supported. If the specified value contains more than one character, only the first one is used.

{% include [libsvm-note-restriction-delimiter-separated-format](../_includes/work_src/reusage-formats/note-restriction-delimiter-separated-format.md) %}

**Possible types**

{% include [reusage-python-cpu-and-gpu](../_includes/work_src/reusage-python/cpu-and-gpu.md) %}

**Default value**

{{ fit__delimiter }}


### has_header

#### Description

Read the column names from the first line of the dataset description file if this parameter is set.

{% include [libsvm-note-restriction-delimiter-separated-format](../_includes/work_src/reusage-formats/note-restriction-delimiter-separated-format.md) %}

**Possible types**

{{ python-type--bool }}

**Default value**

False

### weight

#### Description

The weight of each object in the input data in the form of a one-dimensional array-like data.

By default, it is set to 1 for all objects.

{% include [group-params-python__group-and-group-weight__restriction](../_includes/work_src/reusage/python__group-and-group-weight__restriction.md) %}

**Possible types**

- {{ python-type--list }}
- {{ python-type--numpy-ndarray }}

**Default value**

None

### group_weight

#### Description

The weights of all objects within the defined groups from the input data in the form of one-dimensional array-like data.

Used for calculating the final values of trees. By default, it is set to 1 for all objects in all groups.

{% include [group-params-python__group-and-group-weight__restriction](../_includes/work_src/reusage/python__group-and-group-weight__restriction.md) %}

**Possible types**

- {{ python-type--list }}
- {{ python-type--numpy-ndarray }}

**Default value**

None

### group_id

#### Description

{% include [sections-with-methods-desc-python__group-id__basic-short-desc](../_includes/work_src/reusage/python__group-id__basic-short-desc.md) %}


{% note warning %}

All objects in the dataset must be grouped by group identifiers if they are present. I.e., the objects with the same group identifier should follow each other in the dataset.

{% cut "Example" %}

For example, let's assume that the dataset consists of documents $d_{1}, d_{2}, d_{3}, d_{4}, d_{5}$. The corresponding groups are $g_{1}, g_{2}, g_{3}, g_{2}, g_{2}$, respectively. The feature vectors for the given documents are $f_{1}, f_{2}, f_{3}, f_{4}, f_{5}$ respectively. Then the dataset can take the following form:

$\begin{pmatrix} d_{2}&g_{2}&f_{2}\\ d_{4}&g_{2}&f_{4}\\ d_{5}&g_{2}&f_{5}\\ d_{3}&g_{3}&f_{3}\\ d_{1}&g_{1}&f_{1} \end{pmatrix}$

The grouped blocks of lines can be input in any order. For example, the following order is equivalent to the previous one:

$\begin{pmatrix} d_{1}&g_{1}&f_{1}\\ d_{3}&g_{3}&f_{3}\\ d_{2}&g_{2}&f_{2}\\ d_{4}&g_{2}&f_{4}\\ d_{5}&g_{2}&f_{5} \end{pmatrix}$

{% endcut %}

{% endnote %}

**Possible types**

- {{ python-type--list }}
- {{ python-type--numpy-ndarray }}

**Default value**

None

### subgroup_id

#### Description

Subgroup identifiers for all input objects. Supported identifier types are:
- {{ python-type--int }}
- string types ({{ python-type--string }} or {{ python-type__unicode }} for Python 2 and {{ python-type__bytes }} or {{ python-type--string }} for Python 3).

**Possible types**

- {{ python-type--list }}
- {{ python-type--numpy-ndarray }}

**Default value**

{{ fit__python__subgroup_id }}

### pairs_weight

#### Description

The weight of each input pair of objects in the form of one-dimensional array-like pairs. The number of given values must match the number of specified pairs.

{% include [reusage-learn_pairs__where_is_used](../_includes/work_src/reusage/learn_pairs__where_is_used.md) %}


By default, it is set to 1 for all pairs.

**Possible types**

- {{ python-type--list }}
- {{ python-type--numpy-ndarray }}

**Default value**

{{ fit__python__pairs-weight }}

### baseline

#### Description

Array of formula values for all input objects. The training starts from these values for all input objects instead of starting from zero.

**Possible types**

- {{ python-type--list }}
- {{ python-type--numpy-ndarray }}

**Default value**

None

### timestamp

#### Description

Timestamps for all input objects.
Should contain non-negative integer values.
Useful for sorting a learning dataset by this field during training.

**Possible types**

- {{ python-type--list }}
- {{ python-type--numpy-ndarray }}

**Default value**

None

### feature_names

#### Description

A list of names for each feature in the dataset.

**Possible types**

{{ python-type--list }}

**Default value**

None

### thread_count

#### Description

The number of threads to use when reading data from file.

Use only when the dataset is read from an input file.


**Possible types**

{{ python-type--int }}

**Default value**

{{ fit__thread_count__wrappers }}

{% include [python__log-params](../_includes/work_src/reusage-python/python__log-params.md) %}


## {{ dl--attributes }} {#attributes}

**{{ python__params-table__title__attribute }}:** [](python-reference_pool_attributes.md)

**{{ python__params-table__title__description }}:**  Return the shape of the dataset.


**{{ python__params-table__title__attribute }}:** [](python-reference_pool_attributes.md)

**{{ python__params-table__title__description }}:**

{% include [is_empty_-is_empty__short-desc](../_includes/work_src/reusage-attributes/is_empty__short-desc.md) %}



## {{ dl--methods }} {#methods}

**Method:** [get_baseline](python-reference_pool_get_baseline.md)

#### Description

{% include [get-baseline-get_baseline-desc](../_includes/work_src/reusage-python/get_baseline-desc.md) %}

**Method:** [get_cat_feature_indices](python-reference_pool_get_cat_feature_indices.md)

#### Description

{% include [get-cat-feature-indices-get_cat_feature_indices-desc](../_includes/work_src/reusage-python/get_cat_feature_indices-desc.md) %}

**Method:** [get_embedding_feature_indices](python-reference_pool_get_embedding_feature_indices.md)

#### Description

{% include [get-embedding-feature-indices-get_embedding_feature_indices-desc](../_includes/work_src/reusage-python/get_embedding_feature_indices-desc.md) %}

**Method:** [get_features](python-reference_pool_get_features.md)

#### Description

{% include [get-features-get_features-desc](../_includes/work_src/reusage-python/get_features-desc.md) %}

**Method:** [get_group_id](python-reference_pool_get_group_id.md)

#### Description

{% include [get_group_id-get_group_id__desc](../_includes/work_src/reusage-python/get_group_id__desc.md) %}

**Method:** [get_label](python-reference_pool_get_label.md)

#### Description

{% include [get-label-get_label-desc](../_includes/work_src/reusage-python/get_label-desc.md) %}

**Method:** [get_text_feature_indices](python-reference_pool_get_text_feature_indices.md)

#### Description

{% include [get_text_feature_indices-get_text_features_indices__desc](../_includes/work_src/reusage-python/get_text_features_indices__desc.md) %}

**Method:** [get_weight](python-reference_pool_get_weight.md)

#### Description

{% include [get_weight-get_weight-desc](../_includes/work_src/reusage-python/get_weight-desc.md) %}

**Method:** [is_quantized](python-reference_pool_is_quantized.md)

#### Description

{% include [pool__is_quantized-python__pool__is_quantized__desc__div](../_includes/work_src/reusage-python/python__pool__is_quantized__desc__div.md) %}

**Method:** [num_col](python-reference_pool_num_col.md)

#### Description

{% include [num-row-and-num-col-num_col-desc](../_includes/work_src/reusage-python/num_col-desc.md) %}

**Method:** [num_row](python-reference_pool_num_row.md)

#### Description

{% include [num-row-and-num-col-num_row-desc](../_includes/work_src/reusage-python/num_row-desc.md) %}

**Method:** [quantize](python-reference_pool_quantized.md)

#### Description

{% include [quantized-python_quantized](../_includes/work_src/reusage-python/python_quantized.md) %}

**Method:** [save](python-reference_pool_save.md)

#### Description

{% include [pool_save-python__pool__save__desc__div](../_includes/work_src/reusage-python/python__pool__save__desc__div.md) %}


**Method:** [save_quantization_borders](python-reference_save_quantization_borders.md)

#### Description

{% include [pool__save_quantization_borders-pool__save_quantization_borders_div](../_includes/work_src/reusage-python/pool__save_quantization_borders_div.md) %}

**Method:** [set_baseline](python-reference_pool_set_baseline.md)

#### Description

{% include [set_baseline-set_baseline__desc](../_includes/work_src/reusage-python/set_baseline__desc.md) %}

**Method:** [set_feature_names](python-reference_pool_set_feature_names.md)

#### Description

{% include [set_feature_names-set_feature_names__desc](../_includes/work_src/reusage-python/set_feature_names__desc.md) %}

**Method:** [set_group_id](python-reference_pool_set_group_id.md)

#### Description

{% include [set_group_id-set_group_id__desc](../_includes/work_src/reusage-python/set_group_id__desc.md) %}


**Method:** [set_group_weight](python-reference_pool_set_group_weight.md)

#### Description

{% include [set_group_weight-set_group_weight__desc](../_includes/work_src/reusage-python/set_group_weight__desc.md) %}

**Method:** [set_pairs](python-reference_pool_set_pairs.md)

#### Description

{% include [set_pairs-set_pairs__desc](../_includes/work_src/reusage-python/set_pairs__desc.md) %}


**Method:** [set_pairs_weight](python-reference_pool_set_pairs_weight.md)

#### Description

{% include [set_pairs_weight-set_pairs_weight__desc](../_includes/work_src/reusage-python/set_pairs_weight__desc.md) %}

**Method:** [set_subgroup_id](python-reference_pool_set_subgroup_id.md)

#### Description

{% include [set_subgroup_id-set_subgroup_identifiers__desc](../_includes/work_src/reusage-python/set_subgroup_identifiers__desc.md) %}


**Method:** [set_weight](python-reference_pool_set_weight.md)

#### Description

{% include [set_weight-set_weight__desc](../_includes/work_src/reusage-python/set_weight__desc.md) %}

**Method:** [slice](python-reference_pool_slice.md)

#### Description

{% include [slice-python__pool__slice__desc](../_includes/work_src/reusage-python/python__pool__slice__desc.md) %}

## {{ dl__usage-examples }} {#usage-examples}

#### Load the dataset using [Pool](../concepts/python-reference_pool.md), train it with [CatBoostClassifier](../concepts/python-reference_catboostclassifier.md) and make a prediction

```python
from catboost import CatBoostClassifier, Pool

train_data = Pool(data=[[1, 4, 5, 6],
                        [4, 5, 6, 7],
                        [30, 40, 50, 60]],
                  label=[1, 1, -1],
                  weight=[0.1, 0.2, 0.3])

model = CatBoostClassifier(iterations=10)

model.fit(train_data)
preds_class = model.predict(train_data)

```
