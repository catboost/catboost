# FAQ

## Why is the metric value on the validation dataset sometimes better than the one on the training dataset? {#metric-on-validation-vs-train}

This happens because auto-generated numerical features that are based on categorical features are calculated differently for the training and validation datasets:
- Training dataset: the feature is calculated differently for every object in the dataset. For each _i_-th object the feature is calculated based on data from the first _i-1_ objects (the first _i-1_ objects in some random permutation).
- Validation dataset: the feature is calculated equally for every object in the dataset. For each object the feature is calculated using data from all objects of the training dataset.

{% include [faq-calculating-data-from-all-objects](../_includes/work_src/reusage-common-phrases/calculating-data-from-all-objects.md) %}

Thus, the loss value on the validation dataset might be better than the loss value for the training dataset, because the validation dataset has more powerful features.

{% cut "Details of the algorithm and the rationale behind this solution" %}

**[{{ ext__papers____arxiv_org_abs_1706_09516__name }}]({{ ext__papers____arxiv_org_abs_1706_09516 }})**

_Liudmila Prokhorenkova, Gleb Gusev, Aleksandr Vorobev, Anna Veronika Dorogush, Andrey Gulin. NeurIPS, 2018_

NeurIPS 2018 paper with explanation of Ordered boosting principles and ordered categorical features statistics.

**[CatBoost: gradient boosting with categorical features support](http://learningsys.org/nips17/assets/papers/paper_11.pdf)**

_Anna Veronika Dorogush, Vasily Ershov, Andrey Gulin. Workshop on ML Systems at NIPS 2017_

A paper explaining the {{ product }} working principles: how it handles categorical features, how it fights overfitting, how GPU training and fast formula applier are implemented.

**Ordered boosting and categorical features processing in {{ product }} short overview**

  {% include [videos-unbiased-boosting-with-cat-features-video](../_includes/work_src/reusage-common-phrases/unbiased-boosting-with-cat-features-video.md) %}

{% endcut %}

## Why can metric values on the training dataset that are output during training, be different from ones output when using model predictions? {#applying-the-model-to-train-dataset}

This happens because auto-generated numerical features that are based on categorical features are calculated differently when training and applying the model.

During the training the feature is calculated differently for every object in the training dataset. For each _i_-th object the feature is calculated based on data from the first _i-1_ objects (the first _i-1_ objects in some random permutation). During the prediction the same feature is calculated using data from all objects from the training dataset.

{% include [faq-calculating-data-from-all-objects](../_includes/work_src/reusage-common-phrases/calculating-data-from-all-objects.md) %}

Thus, the loss value calculated during for the prediction might be better than the one that is printed out during the training even though the same dataset is used.

{% cut "Details of the algorithm and the rationale behind this solution" %}

**[{{ ext__papers____arxiv_org_abs_1706_09516__name }}]({{ ext__papers____arxiv_org_abs_1706_09516 }})**

_Liudmila Prokhorenkova, Gleb Gusev, Aleksandr Vorobev, Anna Veronika Dorogush, Andrey Gulin. NeurIPS, 2018_

NeurIPS 2018 paper with explanation of Ordered boosting principles and ordered categorical features statistics.

**[CatBoost: gradient boosting with categorical features support](http://learningsys.org/nips17/assets/papers/paper_11.pdf)**

_Anna Veronika Dorogush, Vasily Ershov, Andrey Gulin. Workshop on ML Systems at NIPS 2017_

A paper explaining the {{ product }} working principles: how it handles categorical features, how it fights overfitting, how GPU training and fast formula applier are implemented.

**Ordered boosting and categorical features processing in {{ product }} short overview**

  {% include [videos-unbiased-boosting-with-cat-features-video](../_includes/work_src/reusage-common-phrases/unbiased-boosting-with-cat-features-video.md) %}

{% endcut %}

## How should weights or baseline be specified for the validation dataset? {#specify-weights-baseline-for-eval-set}

Use the [Pool](python-reference_pool.md) class.

An example of specifying weights:
```python
from catboost import CatBoostClassifier, Pool

train_data = Pool(
    data=[[1, 4, 5, 6],
          [4, 5, 6, 7],
          [30, 40, 50, 60]],
    label=[1, 1, -1],
    weight=[0.1, 0.2, 0.3]
)

eval_data = Pool(
    data=[[1, 4, 5, 6],
          [4, 5, 6, 7],
          [30, 40, 50, 60]],
    label=[1, 0, -1],
    weight=[0.7, 0.1, 0.3]
)

model = CatBoostClassifier(iterations=10)

model.fit(X=train_data, eval_set=eval_data)

```

## Why is it forbidden to use float values and nan values for categorical features? {#why-float-and-nan-values-are-forbidden-for-cat-features}

The algorithm should work identically regardless of the input data format (file or matrix). If the dataset is read from a file all values of categorical features are treated as strings. To treat it the same way when training from matrix, a unique string representation of each feature value is required. There is no unique string representation for floating point values and for nan values.

## Floating point values

If floating point categorical features are allowed the following problem arises.

The feature _f_ is categorical and takes values <q>1</q> and <q>2</q>.

A matrix is used for the training. The column that corresponds to the feature _f_ contains values <q>1.0</q> and <q>2.0</q>.

Each categorical feature value is converted to a string during the training to calculate the corresponding hash value. <q>1.0</q> is converted to the string <q>1.0</q> , and <q>2.0</q> is converted to the string <q>2.0</q>.

After the training the prediction is performed on file.

The column with the feature <q>f</q> contains values <q>1</q> and <q>2</q>.

During the prediction, the hash value of the string <q>1</q> is calculated. This value is not equal to the hash value of the string <q>1.0</q>.

{% include [faq-the-model-doesntcollate](../_includes/work_src/reusage-common-phrases/the-model-doesntcollate.md) %}

## None categorical feature

The feature _f_ is categorical and takes the value <q>None</q> for some object _Obj_.

A matrix is used for the training. The column that contains the value of the feature _f_ for the object _Obj_ contains the value <q>None</q>.

Each categorical feature value is converted to a string during the training to calculate the corresponding hash value. The <q>None</q> value is converted to the string <q>None</q>.

After the training the prediction is performed on file. The column with the feature _f _ contains the value <q>N/A</q>, which would be parsed as <q>None</q> if it was read to a {{ python-type--pandasDataFrame }} before the training.

The hash value of the string <q>N/A</q> is calculated during the prediction. This value is not equal to the hash value of the string <q>None</q>.

{% include [faq-the-model-doesntcollate](../_includes/work_src/reusage-common-phrases/the-model-doesntcollate.md) %}

Since it is not possible to guarantee that the string representation of floating point and None values are the same when reading data from a file or converting the value to a string in Python or any other language, it is required to use strings instead of floating point and None values.


## How to use GridSearchCV and RandomSearchCV from sklearn with categorical features? {#grid-search-random-search-cv}

Use the `cat_features`parameter when constructing the model ([CatBoost](python-reference_catboost.md), [CatBoostRegressor](python-reference_catboostregressor.md) or [CatBoostClassifier](python-reference_catboostclassifier.md)).

Example:
```python
model = catboost.CatBoostRegressor(cat_features=[0,1,2]) grid_search =
            sklearn.model_selection.GridSearchCV(model,
    param_grid)
```


## How to understand which categorical feature combinations have been selected during the training? {#which-cat-feature-combinations-selected-during-training}

Use the [{{ title__internal-feature-importance }}](fstr.md#internal-feature-importance) to familiarize with the resulting combinations. Generate this file from the [command-line](cli-reference_fstr-calc.md) by setting the `--fstr-type` parameter to {{ title__internal-feature-importance }}.

The format of the resulting file is described [here](output-data_feature-analysis_feature-importance.md).

The default feature importances are calculated in accordance with the following principles:
1. Importances of all numerical features are calculated. Some of the numerical features are auto-generated based on categorical features and feature combinations.
1. These importances are shared between initial features. If a numerical feature is auto-generated based on a feature combination, then the importance value is shared equally between the combination participants.

The file that is generate in the {{ title__internal-feature-importance }} mode contains the description of initial numerical features and their importances.


## How to overcome the <q>Out of memory</q> error when training on GPU? {#overcome-out-of-memory}

- Set the `--boosting-type` for the Command-line version parameter to {{ fit__boosting-type__plain }}. It is set to {{ fit__boosting-type__ordered }} by default for datasets with less then 50 thousand objects. The{{ fit__boosting-type__ordered }} scheme requires a lot of memory.
- Set the `--max-ctr-complexity` for the Command-line version parameter to either 1 or 2 if the dataset has categorical features.
- Decrease the value of the `--gpu-ram-part` for the Command-line version parameter.
- Set the `--gpu-cat-features-storage` for the Command-line version parameter to {{ fit__gpu__gpu_cat_features_storage__value__CpuPinnedMemory }}.
- Check that the dataset fits in GPU memory. The quantized version of the dataset is loaded into GPU memory. This version is much smaller than the initial dataset. But it can exceed the available memory if the dataset is large enough.
- Decrease the depth value, if it is greater than 10. Each tree contains $2^{n}$ leaves if the depth is set to $n$, because {{ product }} builds full symmetric trees by default. The recommended depth is 6, which works well in most cases. In rare cases it's useful to increase the depth value up to 10.


## How to reduce the size of the final model? {#reduce-the-size-of-the-final-model}

If the dataset contains categorical features with many different values, the size of the resulting model may be huge. Try the following approaches to reduce the size of the resulting model:

- Decrease the `--max-ctr-complexity` for the Command-line version to either 1 or 2
- For training on CPU:
    - Increase the value of the `--model-size-reg` for the Command-line version parameter.
    - Set the value of the `--ctr-leaf-count-limit` for the Command-line version parameter. {{ fit--ctr-leaf-count-limit }} be default.

- Decrease the value of the `--iterations` for the Command-line version parameter and increase the value of the `--learning-rate` for the Command-line version parameter.
- Remove categorical features that have a small feature importance from the training dataset.


## How to get the model with best parameters from the python cv function? {#how-to-get-best-parameters-from-the-cv-function}

It is not possible. The {{ product }} cv function is intended for cross-validation only, it can not be used for tuning parameter.

The dataset is split into _N_ folds. _N–1_ folds are used for training and one fold is used for model performance estimation. At each iteration, the model is evaluated on all _N_ folds independently. The average score with standard deviation is computed for each iteration.

The only parameter that can be selected based on cross-validation is the number of iterations. Select the best iteration based on the information of the cv results and train the final model with this number of iterations.


## What are the differences between training on CPU and GPU? {#differencies-between-cpu-and-gpu}

- The default value of the `--border-count` for the Command-line version parameter depends on the processing unit type and other parameters:
    - {{ calcer_type__cpu }}: 254
    - {{ calcer_type__gpu }} in {{ error-function__PairLogitPairwise }} and {{ error-function__YetiRankPairwise }} modes: 32
    - {{ calcer_type__gpu }} in all other modes: 128

- Training on {{ calcer_type__cpu }} has the `model_size_reg` set by default. It decreases the size of models that have categorical features. This option is turned off for training on GPU.
- Training on {{ calcer_type__gpu }} is non-deterministic, because the order of floating point summations is non-deterministic in this implementation.
- The following parameters are not supported if training is performed on {{ calcer_type__gpu }}: `--ctr-leaf-count-limit` for the Command-line version, `--monotone-constraints` for the Command-line version.
- The default value of the `--leaf-estimation-method` for the {{ error-function--Quantile }} and {{ error-function--MAE }} loss functions is {{ fit__leaf_estimation_method__Exact }} on {{ cpu-gpu }}.
- Combinations of categorical features are not supported for the following modes if training is performed on {{ calcer_type__gpu }}: {{ error-function--MultiClass }} and {{ error-function--MultiClassOneVsAll }}. The default value of the `--max-ctr-complexity` for the Command-line version parameter for such cases is set to 1.
- The default values for the following parameters depend on the processing unit type:
    - `--bootstrap-type` for the Command-line version:

    {% include [reusage-default-values-bootstrap_type__div](../_includes/work_src/reusage-default-values/bootstrap_type__div.md) %}

    - `--boosting-type` for the Command-line version:

    {% include [reusage-default-values-boosting-type-div](../_includes/work_src/reusage-default-values/boosting-type-div.md) %}

- `--model-size-reg` for the Command-line version:

    {% include [model-reg-size-model-size-reg-gpu-vs-cpu](../_includes/work_src/reusage-common-phrases/model-size-reg-gpu-vs-cpu.md) %}

    Refer to the [Model size regularization coefficient](../references/model-size-reg.md) section for details on the calculation principles.


## Does {{ product }} require preprocessing of missing values? {#preprocessing-of-missing-values}

{{ product }} can handle missing values internally. None values should be used for missing value representation.

If the dataset is read from a file, missing values can be represented as strings like N/A, NAN, None, empty string and the like.

Refer to the [{{ title__missing-values-processing }}](algorithm-missing-values-processing.md) section for details.
