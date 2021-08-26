# catboost.save_pool

```no-highlight
catboost.save_pool(data,
                   label = NULL,
                   weight = NULL,
                   baseline = NULL,
                   pool_path = "data.pool",
                   cd_path = "cd.pool")
```

## {{ dl--purpose }} {#purpose}

{% include [reusage-r-save_pool__purpose](../_includes/work_src/reusage-r/save_pool__purpose.md) %}


Use the [catboost.load_pool](r-reference_catboost-load_pool.md) function to read the resulting files. These files can also be used in the  [Command-line version](cli-installation.md) and the [{{ python-package }}](python-installation.md).

## {{ dl--args }} {#arguments}
###  data

#### Description
A data.frame or matrix with features.

The following column types are supported:
- {{ r-types--double }}
- {{ r-types--factor }}. It is assumed that categorical features are given in this type of columns. A standard {{ product }} processing procedure is applied to this type of columns:
    1. The values are converted to strings.
    1. The `ConvertCatFeatureToFloat` function is applied to the resulting string.


**Default value**

 {{ r--required }}

### label

#### Description
{% include [r-r__reference__label__dataset-not-training__short-desc](../_includes/work_src/reusage/r__reference__label__dataset-not-training__short-desc.md) %}



**Default value**

 NULL

###  weight

#### Description The weights of objects.

**Default value**

 NULL

### baseline

#### Description
A vector of formula values for all input objects. The training starts from these values for all input objects instead of starting from zero.


**Default value**

 NULL

### pool_path

#### Description
The path to the output file that contains the [dataset description](input-data_values-file.md).

**Default value**

 data.pool

### cd_path

#### Description

The path to the output file that contains the [columns description](input-data_column-descfile.md).

**Default value**

 cd.pool
