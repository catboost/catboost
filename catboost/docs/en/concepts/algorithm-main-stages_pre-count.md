# Preliminary calculation of splits

[Quantization](quantization.md) is performed for each numerical feature to determine the possible ways to _split_ data into _buckets_. The resulting information is used for [choosing the tree structure](algorithm-main-stages_choose-tree-structure.md).

The quantization method and number of buckets are set in the starting parameters.

