# How training is performed

The goal of training is to select the _model_ $y$, depending on a set of _features_ $x_{i}$, that best solves the given problem (regression, classification, or multiclassification) for any input _object_. This model is found by using a _training dataset_, which is a set of objects with known features and label values. Accuracy is checked on the _validation dataset_, which has data in the same format as in the training dataset, but it is only used for evaluating the quality of training (it is not used for training).

{{ product }} is based on gradient boosted decision trees. During training, a set of decision trees is built consecutively. Each successive tree is built with reduced loss compared to the previous trees.

The number of trees is controlled by the starting parameters. To prevent overfitting, use the [overfitting detector](overfitting-detector.md). When it is triggered, trees stop being built.

Building stages for a single tree:
1. [Preliminary calculation of splits](algorithm-main-stages_pre-count.md).
1. (_Optional_) [Transforming categorical features to numerical features](algorithm-main-stages_cat-to-numberic.md).
1. (_Optional_) [Transforming text features to numerical features](algorithm-main-stages_text-to-numeric.md).
1. (_Optional_) [Transforming embedding features to numerical features](algorithm-main-stages_embedding-to-numeric.md).
1. [Choosing the tree structure](algorithm-main-stages_choose-tree-structure.md). This stage is affected by the set [Bootstrap options](algorithm-main-stages_bootstrap-options.md).
1. Calculating values in leaves.
