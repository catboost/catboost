# create_cd

{% include [utils-utils__create_cd__desc](../_includes/work_src/reusage-python/utils__create_cd__desc.md) %}


## {{ dl--invoke-format }} {#call-format}

```python
create_cd(label=None,
          cat_features=None,
          text_features=None,
          embedding_features=None,
          weight=None,
          baseline=None,
          doc_id=None,
          group_id=None,
          subgroup_id=None,
          timestamp=None,
          auxiliary_columns=None,
          feature_names=None,
          output_path='train.cd')
```

## {{ dl--parameters }} {#parameters}

### label

#### Description

A zero-based index of the column that defines the target variable (in other words, the object's label value).

**Possible types**

{{ python-type--int }}

**Default value**

None

### cat_features

#### Description

Zero-based indices of columns that define categorical features.

**Possible types**

- {{ python-type--int }}
- {{ python-type__list_of_int }}

**Default value**

None

### text_features

#### Description

Zero-based indices of columns that define text features.

**Possible types**

- {{ python-type--int }}
- {{ python-type__list_of_int }}

**Default value**

None

### embedding_features

#### Description

Zero-based indices of columns that define embedding features.

**Possible types**

- {{ python-type--int }}
- {{ python-type__list_of_int }}

**Default value**

None

### weight

#### Description

A zero-based index of the column that defines the object's weight.

**Possible types**

{{ python-type--int }}

**Default value**

None

### baseline

#### Description

A zero-based index of the column that defines the initial formula values for all input objects.

**Possible types**

{{ python-type--int }}

**Default value**

None

### doc_id

#### Description

A zero-based index of the column that defines the alphanumeric ID of the object.

**Possible types**

{{ python-type--int }}

**Default value**

None

### group_id

#### Description

A zero-based index of the column that defines the identifier of the object's group.

**Possible types**

{{ python-type--int }}

**Default value**

None

### subgroup_id

#### Description

A zero-based index of the column that defines the identifier of the object's subgroup.

**Possible types**

{{ python-type--int }}

**Default value**

None

### timestamp

#### Description

A zero-based index of the column that defines the timestamp of the object.

**Possible types**

{{ python-type--int }}

**Default value**

None

### auxiliary_columns

#### Description

Zero-based indices of columns that define arbitrary data.

**Possible types**

- {{ python-type--int }}
- {{ python-type__list_of_int }}

**Default value**

None

### feature_names

#### Description

A dictionary with the list of column indices and the corresponding feature names.

**Possible types**

{{ python-type--dict }}

For example, use the `feature_names` dictionary to set the names of features in the columns indexed as 4, 5 and 12:
```python
feature_names = {
    4: 'Categ1',
    5: 'Categ2',
    12: 'Num1'
}
```
**Default value**

None


### output_path

#### Description

The path to the output file with columns description.

**Possible types**

{{ python-type--string }}

**Default value**

train.cd



{% note info %}

A parameter for creating columns of the {{ cd-file__col-type__Num }} type is not provided, because columns that contain numerical features don't require descriptions.

{% endnote %}


## {{ dl__usage-examples }} {#usage-examples}

```python
from catboost.utils import create_cd
feature_names = {
    4: 'Categ1',
    5: 'Categ2',
    12: 'Num1'
}

create_cd(
    label=0,
    cat_features=(4, 5, 6),
    weight=1,
    baseline=2,
    doc_id=3,
    group_id=7,
    subgroup_id=8,
    timestamp=9,
    auxiliary_columns=(10, 11),
    feature_names=feature_names,
    output_path='train.cd'
)
```

