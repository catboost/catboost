# Text and embedding features in native Metal training

Native `CatBoost(..., task_type="GPU")` training uses CatBoost's shared text and
embedding estimators to prepare derived numeric features, then searches and
fits trees on Metal. This is the same division of work used by the CUDA
estimated-feature implementation. No CPU CatBoost model is fitted.

The native adapter supports the shared text estimators `BoW`, `NaiveBayes` and
`BM25`, and embedding estimators `LDA` and `KNN`. Shared CatBoost validation
continues to select estimators appropriate to the objective: BoW needs no
target; NaiveBayes and BM25 need classification targets; embedding estimators
support classification and regression targets.

```python
from catboost import CatBoost, Pool

pool = Pool([["clear sunny"], ["dark rainy"], ["clear warm"], ["dark cold"]],
            label=[1, 0, 1, 0], text_features=[0])
model = CatBoost({
    "task_type": "GPU",
    "loss_function": "Logloss",
    "text_processing": {
        "tokenizers": [{"tokenizer_id": "Space", "delimiter": " "}],
        "dictionaries": [{"dictionary_id": "Word", "occurrence_lower_bound": "1"}],
        "feature_processing": {"default": [{
            "tokenizers_names": ["Space"],
            "dictionaries_names": ["Word"],
            "feature_calcers": ["BoW", "NaiveBayes"],
        }]},
    },
})
model.fit(pool)
model.save_model("text.cbm")
```

The shared Python `Pool` API accepts `embedding_features` and model options
accept `embedding_calcers`, for example `["LDA", "KNN:k=3"]`. Text and
embedding source columns may appear together with numeric and categorical
columns. Exported CBM models retain original feature positions/names and
include the trimmed final processing collections used by normal prediction.

## Training histories and quantization

Offline estimators are computed once and copied into each training bank.
Online estimators compute each row from the preceding history, using the same
original-row permutations and group order as its Metal CTR bank and boosting
cursor. Online estimators require permutation banks even when there are no
categorical features. Plain DocParallel, Plain FeatureParallel and Ordered
FeatureParallel use their established history rules; supported greedy tree
policies share the DocParallel banks.

All permutations use borders computed from learn permutation zero. These are
the shared global float binarization settings, with at most 255 borders. BoW
columns use the literal border `0.5`; constant scalar estimated columns retain
CUDA's `0.5` border. Estimated values must be finite. Allocation metadata keeps
the original estimator bin-count hints separately from its computed borders.

Estimated features do not become CTR source categories. A tree tensor that
contains an estimated split cannot produce a compound CTR, matching the CUDA
feature manager. Other eligible categorical tensors remain available.

## Evaluation and snapshots

Each transient tree receives its own trimmed final calcers before metric
updates. Training evaluation pools are already quantized, so the Metal
progress adapter explicitly computes final text calcers from their tokenized
source holders and embedding calcers from their embedding holders. It combines
those estimated buckets with shared numeric, one-hot and CTR quantization and
uses the normal model reader for the per-tree metric increment. The shared CTR
provider includes estimated buckets when locating a one-hot predicate inside a
compound CTR. This avoids mistaking online learn features for full-learn
evaluation features or hashing the wrong bucket in a mixed model.

Snapshots fingerprint source text tokens and embedding rows, stable estimator
indices, borders and computed feature values/bins. Randomly generated estimator
GUIDs are excluded. Equivalent data reconstructs the same feature registry and
histories before saved prediction cursors and tree split indices are restored.
Existing numeric/categorical snapshot checksums remain unchanged.

## Current boundaries

The standalone `MetalCatBoost*` estimators do not expose text or embedding
calcers. Use native CatBoost `Pool` training for these features.

GPU model prediction for text/embedding models retains the CUDA evaluator's
explicit rejection; normal prediction uses an unquantized Pool with the exported shared calcers
and model reader. Prediction from a prequantized Pool is rejected explicitly
because the shared prequantized evaluator does not prepare estimated buckets. Initial-model continuation with text or embedding model metadata
is rejected early: the shared model summation code cannot merge those
processing collections, and CUDA training does not support initial-model
continuation. Supported native numeric/categorical continuation is unchanged.

Shared model export restrictions still apply, including which formats can
represent processing collections. The `final_feature_calcer_computation_mode`
option controls final calcer export; choosing `Skip` deliberately produces a
model without the processing collections required for ordinary prediction.

Implementation: [estimated_features.cpp](../train_lib/estimated_features.cpp),
[estimated_features_apply.cpp](../train_lib/estimated_features_apply.cpp),
[estimated_features_checksum.h](../train_lib/estimated_features_checksum.h).
Focused acceptance cases are in
[test_native_estimated_features.py](../tests/test_native_estimated_features.py).
