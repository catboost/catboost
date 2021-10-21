# Pool initialization

## Loading data from a file {#data-from-file}

A list of possible methods to load the dataset from a file is given in the table below.

### Use the default columns description

If the columns description file is omitted,  it is assumed that the first column in the file with the dataset description defines the label value, and the other columns are the values of numerical features.

Usage example:

```python
Pool(dataset_desc_file)
```

### Use the specified columns description

If specified, the `cd_file` should contain the [columns description](input-data_column-descfile.md).

Usage example:

```python
Pool(dataset_desc_file, column_description=cd_file)
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
