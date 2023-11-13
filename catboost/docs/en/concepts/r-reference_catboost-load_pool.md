# catboost.load_pool

```no-highlight
catboost.load_pool(data,
                   label = NULL,
                   cat_features = NULL,
                   column_description = NULL,
                   pairs = NULL,
                   delimiter = "\t",
                   has_header = FALSE,
                   weight = NULL,
                   group_id = NULL,
                   group_weight = NULL,
                   subgroup_id = NULL,
                   pairs_weight = NULL,
                   baseline = NULL,
                   feature_names = NULL,
                   thread_count = -1)
```

## {{ dl--purpose }} {#purpose}

{% include [reusage-r-load_pool__purpose](../_includes/work_src/reusage-r/load_pool__purpose.md) %}


## {{ dl--args }} {#arguments}
### data

#### Description

A file path, data.frame or matrix with features.

{% include [r-data__column-types-descc](../_includes/work_src/reusage/data__column-types-descc.md) %}

{% if audience == "internal" %}

#### For datasets input as files

{% include [files-internal-files-internal__desc__full](../yandex_specific/_includes/reusage-formats/files-only-internal__desc__full.md) %}

{% endif %}

**Default value**

{{ r--required }}

### label


#### Description


The target variables (in other words, the objects' label values) of the dataset.

{% include [r-r__this-parameter-is-used-if-the-input-data-format-is-matrix](../_includes/work_src/reusage/r__this-parameter-is-used-if-the-input-data-format-is-matrix.md) %}



**Default value**

NULL

### cat_features


#### Description


A vector of categorical features indices.

The indices are zero-based and can differ from the ones given in the [columns description](../concepts/input-data_column-descfile.md) file.

If `data` parameter is `data.frame` don't use `cat_features`, categorical features are determined automatically
 from `data.frame` column types.


**Default value**

{{ fit__r__cat_features }}

### column_description


#### Description


{% include [reusage-cd-short-desct](../_includes/work_src/reusage/cd-short-desct.md) %}


{% if audience == "internal" %}

{% include [internal__cd-internal-cd-desc](../yandex_specific/_includes/reusage-formats/internal-cd-desc.md) %}

{% endif %}

This parameter is used if the data is input from a file.


**Default value**

NULL, it is assumed that the first column in the file with the dataset description defines the label value, and the other columns are the values of numerical features.

### pairs


#### Description


A file path, matrix or data.frame with  pairs description of shape `N` by 2:

- `N` is the number of pairs.
- The first element of the pair is the zero-based index of the winner object from the input dataset for pairwise comparison.
- The second element of the pair is the zero-based index of the loser object from the input dataset for pairwise comparison.

{% include [reusage-learn_pairs__where_is_used](../_includes/work_src/reusage/learn_pairs__where_is_used.md) %}



**Default value**


NULL

{% include [loss-functions-pairwisemetrics_require_pairs_data](../_includes/work_src/reusage-common-phrases/pairwisemetrics_require_pairs_data.md) %}



### delimiter


#### Description


The delimiter character used to separate the data in the dataset description input file.

Only single char delimiters are supported. If the specified value contains more than one character, only the first one is used.

{% include [libsvm-note-restriction-delimiter-separated-format](../_includes/work_src/reusage-formats/note-restriction-delimiter-separated-format.md) %}


**Default value**

\t

### has_header


#### Description


Read the column names from the first line of the dataset description file if this parameter is set.

{% include [libsvm-note-restriction-delimiter-separated-format](../_includes/work_src/reusage-formats/note-restriction-delimiter-separated-format.md) %}



**Default value**

FALSE

### weight


#### Description

The weights of objects.

**Default value**

NULL

### group_id


#### Description


Group identifiers for all input objects.

{% include [methods-param-desc-group-id__desc__group-by-group-id__obligatory__note](../_includes/work_src/reusage/group-id__desc__group-by-group-id__obligatory__note.md) %}


**Default value**

NULL

### group_weight


#### Description


{% include [methods-param-desc-python__group_weight__first-sentence](../_includes/work_src/reusage/python__group_weight__first-sentence.md) %}


{% include [group-params-r__group-and-group-weight__restriction](../_includes/work_src/reusage/r__group-and-group-weight__restriction.md) %}



**Default value**

NULL

### subgroup_id


#### Description


Subgroup identifiers for all input objects.


**Default value**

NULL

### pairs_weight


#### Description


The weight of each input pair of objects.

{% include [reusage-learn_pairs__where_is_used](../_includes/work_src/reusage/learn_pairs__where_is_used.md) %}


{% include [methods-param-desc-python__pairs-weight__default-text](../_includes/work_src/reusage/python__pairs-weight__default-text.md) %}


Do not use this parameter if an input file is specified in the `pairs` parameter.


**Default value**

{{ fit__r__pairs-weight }}

### baseline


#### Description


A vector of formula values for all input objects. The training starts from these values for all input objects instead of starting from zero.


**Default value**

NULL

### feature_names


#### Description


{% include [methods-param-desc-feature_names__desc](../_includes/work_src/reusage/feature_names__desc.md) %}



**Default value**

NULL

### thread_count


#### Description

The number of threads to use while reading the data.
Optimizes the reading time. This parameter doesn't affect the results.


**Default value**

{{ fit__thread_count__wrappers }}

## {{ dl--example }} {#example}

{% include [load-load-from-file](../_includes/work_src/reusage-code-examples/load-from-file.md) %}


{% include [load-load-from-the-package](../_includes/work_src/reusage-code-examples/load-from-the-package.md) %}


{% include [load-load-the-dataset-from-data-frame](../_includes/work_src/reusage-code-examples/load-the-dataset-from-data-frame.md) %}
