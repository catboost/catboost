# ONNX

ONNX is an open format to represent AI models.

A quote from the [Open Neural Network Exchange](https://github.com/onnx/onnx/blob/dc75285d4a1cff9618400164dfdb26c5a1bab70a/docs/IR.md#components) documentation:

<q>There are two official ONNX variants; the main distinction between the two is found in the supported types and the default operator sets. The neural-network-only ONNX variant recognizes only tensors as input and output types, while the Classical Machine Learning extension, ONNX-ML, also recognizes sequences and maps. ONNX-ML extends the ONNX operator set with ML algorithms that are not based on neural networks.</q>

{{ product }} models are based on ensembles of decision trees, therefore only exporting models to the ONNX-ML format is supported.

## Specifics {#specifics}

- Only models trained on datasets without categorical features are currently supported.
- Exported ONNX-ML models cannot be currently loaded and applied by {{ product }} libraries/executable. This export format is suitable only for external Machine Learning libraries.
- {% include [reusage-common-phrases-native-catboost-format-is-faster](../_includes/work_src/reusage-common-phrases/native-catboost-format-is-faster.md) %}

- The model's metadata is stored in the `metadata_props` component of the [ONNX Model](https://github.com/onnx/onnx/blob/master/docs/IR.md#models).

## Applying a trained model with ONNX {#applying-model}

### Model input parameters

#### features

The input features.

Possible types: Tensor of shape [N_examples] and type {{ python-type--int }} or {{ python-type--string }}



### Model output parameters for classification

#### label

The label value for the example.

{% note info %}

The label is inferred incorrectly for binary classification. This is a known bug in the onnxruntime implementation. Ignore the value of this parameter in case of binary classification.

{% endnote %}

Possible types: tensor of shape [N_examples] and one of the following types:
- {{ python-type--int }} if class names are not specified in the training dataset.
- {{ python-type--string }} if class names are specified in the training dataset.


#### probabilities

The key value reflects the probability that the example belongs to the class defined by the map key.

Possible types: tensor of shape [N_examples] and one of the following types:
- type seq(map(string, float)) if class names are specified in the training dataset.
- seq(map(int64, float)) if class names are not specified in the training dataset.



### Model output parameters for regression

#### predictions

The target value predicted by the model.

Possible types: tensor of shape [N_examples] and type float



## Examples {#examples}

The following examples use the [{{ python-package }}](python-quickstart.md) for training and the[ONNX Runtime](https://github.com/Microsoft/onnxruntime) scoring engine for applying the model.

### Binary classification

Train the model with {{ product }}:

```python
import catboost
from sklearn import datasets


breast_cancer = datasets.load_breast_cancer()
model = catboost.CatBoostClassifier(loss_function='Logloss')

model.fit(breast_cancer.data, breast_cancer.target)

# Save model to ONNX-ML format
model.save_model(
    "breast_cancer.onnx",
    format="onnx",
    export_parameters={
        'onnx_domain': 'ai.catboost',
        'onnx_model_version': 1,
        'onnx_doc_string': 'test model for BinaryClassification',
        'onnx_graph_name': 'CatBoostModel_for_BinaryClassification'
    }
)
```

Apply the model with onnxruntime:

```python
import numpy as np
from sklearn import datasets
import onnxruntime as rt


breast_cancer = datasets.load_breast_cancer()

sess = rt.InferenceSession('breast_cancer.onnx')

# onnxruntime bug: 'label' inference is broken for binary classification
#label = sess.run(['label'],
#                 {'features': breast_cancer.data.astype(np.float32)})

probabilities = sess.run(['probabilities'],
                         {'features': breast_cancer.data.astype(np.float32)})
```

### Multiclassification

Train the model with {{ product }}:

```python
import catboost
from sklearn import datasets


iris = datasets.load_iris()
model = catboost.CatBoostClassifier(loss_function='MultiClass')

model.fit(iris.data, iris.target)

# Save model to ONNX-ML format
model.save_model(
    "iris.onnx",
    format="onnx",
    export_parameters={
        'onnx_domain': 'ai.catboost',
        'onnx_model_version': 1,
        'onnx_doc_string': 'test model for MultiClassification',
        'onnx_graph_name': 'CatBoostModel_for_MultiClassification'
    }
)
```

Apply the model with onnxruntime:

```python
import numpy as np
from sklearn import datasets
import onnxruntime as rt


iris = datasets.load_iris()

sess = rt.InferenceSession('iris.onnx')

# can get only label
label = sess.run(['label'],
                 {'features': iris.data.astype(np.float32)})

# can get only probabilities
probabilities = sess.run(['probabilities'],
                         {'features': iris.data.astype(np.float32)})

# or both
label, probabilities = sess.run(['label', 'probabilities'],
                                {'features': iris.data.astype(np.float32)})
```

### Regression

Train the model with {{ product }}:

```python
import catboost
from sklearn import datasets


boston = datasets.load_boston()
model = catboost.CatBoostRegressor()

model.fit(boston.data, boston.target)

# Save model to ONNX-ML format
model.save_model(
    "boston.onnx",
    format="onnx",
    export_parameters={
        'onnx_domain': 'ai.catboost',
        'onnx_model_version': 1,
        'onnx_doc_string': 'test model for Regression',
        'onnx_graph_name': 'CatBoostModel_for_Regression'
    }
)
```

Apply the model with onnxruntime:

```python
import numpy as np
from sklearn import datasets
import onnxruntime as rt


boston = datasets.load_boston()

sess = rt.InferenceSession('boston.onnx')

predictions = sess.run(['predictions'],
                       {'features': boston.data.astype(np.float32)})
```

