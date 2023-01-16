
Return the number of trees in the model.

This number can differ from the value specified in the `--iterations` training parameter in the following cases:
- The training is stopped by the [overfitting detector](../../../concepts/overfitting-detector.md).
- The `--use-best-model` training parameter is set to <q>True</q>.
