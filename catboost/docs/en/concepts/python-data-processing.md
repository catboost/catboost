# Pool initialization

## Loading data from a file {#data-from-file}

A list of possible methods to load the dataset from a file is given in the table below.

### From a file in DSV format with default columns

If the columns description file is omitted,  it is assumed that the first column in the file with the dataset description defines the label value, and the other columns are the values of numerical features.

Usage example:

```python
Pool(dataset_desc_file)
```

### From a file in DSV format with custom columns

If specified, the `cd_file` should contain the [columns description](input-data_column-descfile.md).

Usage example:

```python
Pool(dataset_desc_file, column_description=cd_file)
```

### From a file in libsvm format

To load a file in [libsvm format](https://github.com/cjlin1/libsvm/blob/557d85749aaf0ca83fd229af0f00e4f4cb7be85c/README#L53) specify a `libsvm://` prefix before a file path in the `Pool`'s constructor `data` argument value.

Usage example:

```python
Pool('libsvm://' + dataset_desc_file)
```

## Loading data from array-like structures {#load-from-array-like-structures}

A list of possible methods to load the dataset from array-like structures is given in the table below.

### Use numerical features only

It is assumed that all features are numerical, since `cat_features` are not defined.

Usage example:

```python
df = pd.read_table(TRAIN_FILE)
Pool(data=df.iloc[:, 1:].values, label=df.iloc[:, 0].values)
```
### Use both numerical and categorical features

It is assumed that the list of feature indices specified in the `cat_features` parameter correspond to categorical features. All other features are considered numerical.

Usage example:

```python
df = pd.read_table(TRAIN_FILE)
Pool(data=df.iloc[:, 1:].values, label=df.iloc[:, 0].values, cat_features=[1,2,3])
```
