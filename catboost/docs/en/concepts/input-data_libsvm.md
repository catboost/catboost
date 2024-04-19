# Dataset description in extended libsvm format

#### {{ input_data__title__contains }}

The input dataset in an extended version of the widely used [libsvm format](https://github.com/cjlin1/libsvm/blob/557d85749aaf0ca83fd229af0f00e4f4cb7be85c/README#L53) for sparse datasets.

#### {{ input_data__title__specification }}

Data is stored in the text file in UTF-8 encoding.

Each line describes an object with a label and some features:

```
<label> <feature_index1>:<feature_value1> <feature_index2>:<feature_value2>
```

- Label is a real value or an integer class index (for classification)

- Feature indices are positive integers starting from 1.

    {% note info %}

    When this dataset is loaded and being processed by the {{ product }} API, features' indices are changed to zero-based. For example, the feature indexed 1 in the file changes its' index to 0 in the {{ product }} APIs.

    {% endnote %}

    Feature indices on each line must be specified in ascending order.

- Feature values are integers, real numbers or strings (without spaces, to avoid breaking the format). Integers and real numbers can be used for numerical features, integers and strings can be used for categorical features.

    Strings as feature values are an extension of the original libsvm format.


To specify categorical features, provide the [columns description file](input-data_column-descfile.md) in the following format:

```
0<\t>Label
<feature1_index><\t><feature1_type><\t><feature1 name (optional)>
<feature2_index><\t><feature2_type><\t><feature2 name (optional)>
...
```

- Feature indices start from 1 as in the libsvm file itself.
- Feature type must be <q>Num</q> for numerical features or <q>Categ</q> for categorical features.
- Feature names are optional and can be specified in the third column.

It is not mandatory to describe all features in the columns description file. Feature indices that are not mentioned in the columns description file are assumed to be numerical by default.

If the input dataset contains only numerical features and feature names don't have to be specified, then columns description file can be omitted.

#### {{ input_data__title__example }}

#### Dataset with numerical features only

```
1 1:0.1 3:2.2 4:3
0 2:0.22 3:0.82
0 1:0.02 4:0.61
1 3:0.72 4:0.5
```

#### Dataset in extended libsvm format with categorical features

Dataset file:
```
1 1:0.1 3:small 4:3 5:Male
0 2:0.22 3:small 5:Female
0 1:0.02 4:0.61 5:Female
1 3:large 4:0.5 5:Male
```

The corresponding columns description file:
```
0	Label
1	Num
2	Num
3	Categ
4	Num
5	Categ
```

