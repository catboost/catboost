# Native Metal feature weights

`feature_weights` multiplies the gain used to choose a split. It does not scale
targets, derivatives, curvature, fitted leaves, or exported leaf masses. Native
scalar, vector, greedy, Ordered, and full-matrix trainers receive the same static
feature-weight mapping.

The GPU option map addresses feature-manager IDs directly, following
[`ExpandFeatureWeights`](../../private/libs/options/feature_penalties_options.cpp).
Original numeric and categorical columns consume IDs in input order, including
ignored columns. Text and embedding inputs do not consume original IDs;
estimated columns receive their own IDs. Simple CTRs are registered separately,
in category/configuration order, so they do not inherit their input category's
weight. For example, an original category at ID 0 and numeric column at ID 1 have
their single simple CTR at ID 2. Weighting ID 0 affects a one-hot candidate when
one exists; weighting ID 2 affects that CTR.

Dynamic tree CTRs deliberately use user weight **1**. CUDA's
[`tree_ctr_datasets_visitor.cpp`](../../cuda/methods/tree_ctr_datasets_visitor.cpp)
passes a global weight vector to a scorer that addresses local pack indices.
Those indices can alias unrelated original features. Metal avoids assigning an
unrelated input's weight to a compound CTR. The existing model-size penalty still
applies independently. This is a documented correction to the observed source
mapping, not a claim of NVIDIA execution equivalence for nonuniform weights.

[`test_native_feature_weights.py`](../tests/test_native_feature_weights.py)
checks independent depth-one gains; original and appended simple-CTR IDs;
equal-column score-only effects across vector, query, and full-matrix paths;
compound CTR independence from original category weights; and snapshots/readers.
Every fit requests Metal. CPU prediction is used only to check model readers.
The expanded acceptance cases are collected but await coordinated GPU execution.
