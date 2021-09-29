# Select features

## {{ dl--purpose }}

{% include [select_features-python__select-features__desc](../_includes/work_src/reusage-python/python__select-features__desc.md) %}


## {{ dl__cli__execution-format }} {#execution-format}

```
catboost select-features -f <file path> --features-for-select <comma-separated indices or names> --num-features-to-select <integer>  [optional parameters]
```

## {{ common-text__title__reference__parameters }} {#options}

Except for the options below, the others are the same as in [Train a model](../references/training-parameters/index.md) mode.

### --features-for-select

#### Description

Features which participate in the selection. The following formats are supported: indices, names, index ranges, name ranges. Values are separated by commas, for example: `0,3,5,6,10-15,City,Player1-Player11`.

**{{ cli__params-table__title__default }}**

 {{ python--required }}

**{{ cli__params-table__title__processing-units-type }}**

 {{ python-processing-units-type }}

### --num-features-to-select

#### Description

The number of features to select from the option `--features-for-select`.

**{{ cli__params-table__title__default }}**

 {{ python--required }}

**{{ cli__params-table__title__processing-units-type }}**

 {{ python-processing-units-type }}

### --features-selection-steps

#### Description

The number of times for training the model. Use more steps for more accurate selection.

**{{ cli__params-table__title__default }}**

 1

**{{ cli__params-table__title__processing-units-type }}**

 {{ python-processing-units-type }}


### --features-selection-algorithm

#### Description

The main algorithm is [Recursive Feature Elimination](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.RFE.html) with variable feature importance calculation method:

- `RecursiveByPredictionValuesChange` — the fastest algorithm and the least accurate method (not recommended for ranking losses).
- `RecursiveByLossFunctionChange` — the optimal option according to accuracy/speed balance.
- `RecursiveByShapValues` — the most accurate method.

**{{ cli__params-table__title__default }}**

 RecursiveByShapValues

**{{ cli__params-table__title__processing-units-type }}**

 {{ python-processing-units-type }}


### --shap-calc-type

#### Description

The method of the SHAP values calculations ordered by accuracy:

- `Approximate`
- `Regular`
- `Exact`

Used in RFE based on LossFunctionChange and ShapValues.

**{{ cli__params-table__title__default }}**

 Regular

**{{ cli__params-table__title__processing-units-type }}**

 {{ python-processing-units-type }}

### --train-final-model

#### Description

If specified, then the model with selected features will be trained and saved to `--model-file`.

**{{ cli__params-table__title__default }}**

 False

**{{ cli__params-table__title__processing-units-type }}**

 {{ python-processing-units-type }}


### --features-selection-result-path

#### Description

Path to the file with [selection results in the JSON format](output-data_features-selection.md).

**{{ cli__params-table__title__default }}**

 selection_result.json

**{{ cli__params-table__title__processing-units-type }}**

 {{ python-processing-units-type }}


## {{ dl__usage-examples }} {#usage-examples-select-features}

```python
catboost select-features --learn-set train.csv --test-set test.csv --column-description train.cd --loss-function RMSE --iterations 100 --features-for-select 0-99 --num-features-to-select 10 --features-selection-steps 3 --features-selection-algorithm RecursiveByShapValues --train-final-model
```
