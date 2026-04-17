# set_label

Replace the label (target variable) values for the dataset in place, without rebuilding the Pool.

Useful for reusing large feature data across different targets, weighting schemes, or task types. The Pool's features, categorical metadata, weights, group IDs, and baseline are preserved.

## {{ dl--invoke-format }} {#call-format}

```python
set_label(label)
```

## {{ dl--parameters }} {#parameters}

### label

#### Description

A one-dimensional array-like of numeric label values. The length must match the number of objects in the dataset (`Pool.num_row()`).

**Possible types**

- {{ python-type--list }}
- {{ python-type__np_ndarray }}
- {{ python-type--pandasSeries }}
- Single-column {{ python-type--pandasDataFrame }}
- [polars.Series](https://docs.pola.rs/api/python/stable/reference/series/index.html)
- Single-column [polars.DataFrame](https://docs.pola.rs/api/python/stable/reference/dataframe/index.html)

Multi-column DataFrames (pandas or polars) are rejected — use the `Pool(data, label=2D_array)` constructor for multi-target.

**Default value**

{{ python--required }}

## {{ dl--output-format }} {#return-value}

{{ python-type--pool }}

Returns `self` (the modified Pool object), enabling chained calls.

## Dtype handling

Labels are stored internally as `float32` regardless of input dtype. After `set_label`, `pool.get_label()` returns an array cast back to the dtype you passed in (e.g. `int64` in, `int64` out), matching the Pool constructor. Note that integer values above `2**24` (~16.78M) lose precision in the float32 storage step.

## Supported Pool types

`set_label` works on:

- Pools built from numpy arrays, pandas/polars data structures, and `FeaturesData` objects.
- Pools loaded from a file via `Pool(data="path.tsv", column_description=...)`.
- Pools after `quantize()` — labels live outside the quantized feature storage and are replaced identically.

## Errors

Raises `CatBoostError` when:

- The label length does not match `Pool.num_row()`.
- The label dtype is non-numeric (strings, bytes, object dtype). Reconstruct the Pool for string/categorical targets.
- The label is a 2-D array with more than one column (or a multi-column DataFrame).
- The Pool already has a label with a non-numeric `TargetType` (e.g. a Pool originally built with string labels cannot be switched to numeric via `set_label`).

## Restrictions

- **Single-target only.** Multi-target labels must be set at construction time.
- **Not thread-safe during training.** Do **not** call `set_label()` concurrently with `model.fit()`, `model.score()`, or `model.eval_metrics()` on the same Pool — those release the GIL and read labels without synchronization. This restriction is shared by every existing `Pool.set_*` method.
- **NaN / Inf values are accepted** (mirroring the Pool constructor). Some loss functions reject them at training time.

## {{ input_data__title__example }} {#example}

```python
import numpy as np
from catboost import Pool

train_data = [[76, 'blvd', 41, 50, 7],
              [75, 'today', 57, 0, 48],
              [70, 'letters', 33, 17, 7],
              [72, 'now', 43, 29, 12],
              [60, 'back', 2, 0, 1]]

label_values = [1, 0, 0, 1, 4]

input_pool = Pool(data=train_data,
                  label=label_values,
                  cat_features=[1])

new_labels = [0, 1, 1, 0, 2]
input_pool.set_label(new_labels)

input_pool.get_label()
```

## See also

- [Pool.set_weight](python-reference_pool_set_weight.md) — the sibling setter whose pattern `set_label` mirrors.
- [fit()](python-reference_catboost_fit.md) — passing `y` alongside a `Pool` now routes through `set_label` (see the note in that page).
