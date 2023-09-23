# Overview

These parameters are for the Python package, R package and Command-line version.

{% include [reusage-python-how-aliases-are-applied-intro](../../_includes/work_src/reusage-python/how-aliases-are-applied-intro.md) %}

{% include [installation-nvidia-driver-reqs](../../_includes/work_src/reusage-code-examples/nvidia-driver-reqs.md) %}

## Common parameters

### [loss_function](common.md#loss_function)

Command-line: `--loss-function`

_Alias:_ `objective`

{% include [reusage-loss-function-short-desc](../../_includes/work_src/reusage/loss-function-short-desc.md) %}

### [custom_metric](common.md#custom_metric)

Command-line: `--custom-metric`

{% include [reusage-custom-loss--basic](../../_includes/work_src/reusage/custom-loss--basic.md) %}

### [eval_metric](common.md#eval_metric)

Command-line: `--eval-metric`

{% include [reusage-eval-metric--basic](../../_includes/work_src/reusage/eval-metric--basic.md) %}

### [iterations](common.md#iterations)

Command-line: `-i`, `--iterations`

_Aliases:_ `num_boost_round`, `n_estimators`, `num_trees`

The maximum number of trees that can be built when solving machine learning problems.

### [learning_rate](common.md#learning_rate)

Command-line: `-w`, `--learning-rate`

_Alias:_ `eta`

The learning rate.

Used for reducing the gradient step.

### [random_seed](common.md#random_seed)

Command-line: `-r`, `--random-seed`

_Alias:_`random_state`

The random seed used for training.

### [l2_leaf_reg](common.md#l2_leaf_reg)

Command-line: `--l2-leaf-reg`, `l2-leaf-regularizer`

_Alias:_ `reg_lambda`

Coefficient at the L2 regularization term of the cost function.

### [bootstrap_type](common.md#bootstrap_type)

Command-line: `--bootstrap-type`

[Bootstrap type](../../concepts/algorithm-main-stages_bootstrap-options.md). Defines the method for sampling the weights of objects.

### [bagging_temperature](common.md#bagging_temperature)

Command-line: `--bagging-temperature`

Defines the settings of the Bayesian bootstrap. It is used by default in classification and regression modes.

### [subsample](common.md#subsample)

Command-line: `--subsample`

Sample rate for bagging.

### [sampling_frequency](common.md#sampling_frequency)

Command-line: `--sampling-frequency`

Frequency to sample weights and objects when building trees.

### [sampling_unit](common.md#sampling_unit)

Command-line: `--sampling-unit`

The sampling scheme.

### [mvs_reg](common.md#mvs_reg)

Command-line: `--mvs-reg`

{% include [reusage-cli__mvs-head-fraction__div](../../_includes/work_src/reusage/cli__mvs-head-fraction__div.md) %}

### [random_strength](common.md#random_strength)

Command-line: `--random-strength`

The amount of randomness to use for scoring splits when the tree structure is selected. Use this parameter to avoid overfitting the model.

### [use_best_model](common.md#use_best_model)

Command-line: `--use-best-model`

If this parameter is set, the number of trees that are saved in the resulting model is defined.

### [best_model_min_trees](common.md#best_model_min_trees)

Command-line: `--best-model-min-trees`

{% include [reusage-clii__best-model-min-trees__short-desc](../../_includes/work_src/reusage/clii__best-model-min-trees__short-desc.md) %}

### [depth](common.md#depth)

Command-line: `-n`, `--depth`

_Alias:_ `max_depth`

Depth of the trees.

### [grow_policy](common.md#grow_policy)

Command-line: `--grow-policy`

The tree growing policy. Defines how to perform greedy tree construction.

### [min_data_in_leaf](common.md#min_data_in_leaf)

Command-line: `--min-data-in-leaf`

_Alias:_ `min_child_samples`

The minimum number of training samples in a leaf. {{ product }} does not search for new splits in leaves with samples count less than the specified value.

### [max_leaves](common.md#max_leavescommon.md#)

Command-line: `--max-leaves`

_Alias:_`num_leaves`

The maximum number of leafs in the resulting tree. Can be used only with the {{ growing_policy__Lossguide }} growing policy.

### [ignored_features](common.md#ignored_features)

Command-line: `-I`, `--ignore-features`

{% include [reusage-ignored-feature__common-div](../../_includes/work_src/reusage/ignored-feature__common-div.md) %}

### [one_hot_max_size](common.md#one_hot_max_size)

Command-line: `--one-hot-max-size`

Use one-hot encoding for all categorical features with a number of different values less than or equal to the given parameter value. Ctrs are not calculated for such features.

### [has_time](common.md#has_time)

Command-line: `--has-time`

Use the order of objects in the input data (do not perform random permutations during the [Transforming categorical features to numerical features](../../concepts/algorithm-main-stages_cat-to-numberic.md) and [Choosing the tree structure](../../concepts/algorithm-main-stages_choose-tree-structure.md) stages).

### [rsm](common.md#rsm)

Command-line: `--rsm`

_Alias:_`colsample_bylevel`

Random subspace method. The percentage of features to use at each split selection, when features are selected over again at random.

### [nan_mode](common.md#nan_mode)

Command-line: `--nan-mode`

The method for  [processing missing values](../../concepts/algorithm-missing-values-processing.md) in the input dataset.

### [input_borders](common.md#input_borders)

Command-line: `--input-borders-file`

Load [Custom quantization borders and missing value modes](../../concepts/input-data_custom-borders.md) from a file (do not generate them).

### [output_borders](common.md#output_borders)

Command-line: `--output-borders-file`

Save quantization borders for the current dataset to a file.

### [fold_permutation_block](common.md#fold_permutation_block)

Command-line: `--fold-permutation-block`

Objects in the dataset are grouped in blocks before the random permutations. This parameter defines the size of the blocks.

### [leaf_estimation_method](common.md#leaf_estimation_method)

Command-line: `--leaf-estimation-method`

The method used to calculate the values in leaves.

### [leaf_estimation_iterations](common.md#leaf_estimation_iterations)

Command-line: `--leaf-estimation-iterations`

This parameter regulates how many steps are done in every tree when calculating leaf values.

### [leaf_estimation_backtracking](common.md#leaf_estimation_backtracking)

Command-line: `--leaf-estimation-backtracking`

When the value of the `leaf_estimation_iterations` parameter is greater than 1, {{ product }} makes several gradient or newton steps when calculating the resulting leaf values of a tree.

### [fold_len_multiplier](common.md#fold_len_multiplier)

Command-line: `--fold-len-multiplier`

Coefficient for changing the length of folds.

### [approx_on_full_history](common.md#approx_on_full_history)

Command-line:`--approx-on-full-history`

The principles for calculating the approximated values.

### [class_weights](common.md#class_weights)

Command-line: `--class-weights`

{% include [reusage-class-weights__short-desc-intro](../../_includes/work_src/reusage/class-weights__short-desc-intro.md) %}

### [class_names](common.md#class_names)

Classes names. Allows to redefine the default values when using the {{ error-function--MultiClass }} and {{ error-function--Logit }} metrics.

### [auto_class_weights](common.md#auto_class_weights)

Command-line: `--auto-class-weights`

{% include [reusage-cli__auto-class-weights__div](../../_includes/work_src/reusage/cli__auto-class-weights__div.md) %}

### [scale_pos_weight](common.md#scale_pos_weight)

The weight for class 1 in binary classification. The value is used as a multiplier for the weights of objects from class 1.

### [boosting_type](common.md#boosting_type)

Command-line: `--boosting-type`

Boosting scheme.

### [boost_from_average](common.md#boost_from_average)

Command-line: `--boost-from-average`

Initialize approximate values by best constant value for the specified loss function.

### [langevin](common.md#langevin)

Command-line: `--langevin`

Enables the Stochastic Gradient Langevin Boosting mode.

### [diffusion_temperature](common.md#diffusion_temperature)

Command-line: `--diffusion-temperature`

The diffusion temperature of the Stochastic Gradient Langevin Boosting mode.

### [posterior_sampling](common.md#posterior_sampling)

Command-line: `--posterior-sampling	`

If this parameter is set several options are specified as follows and model parameters are checked to obtain uncertainty predictions with good theoretical properties.

### [allow_const_label](common.md#allow_const_label)

Command-line: `--allow-const-label`

Use it to train models with datasets that have equal label values for all objects.

### [score_function](common.md#score_function)

Command-line: `--score-function`

The [score type](../../concepts/algorithm-score-functions.md) used to select the next split during the tree construction.

### [monotone_constraints](common.md#monotone_constraints)

Command-line: `--monotone-constraints`

{% include [reusage-cli__monotone-constraints__desc__div](../../_includes/work_src/reusage/cli__monotone-constraints__desc__div.md) %}

### [feature_weights](common.md#feature_weights)

Command-line: `--feature-weights`

{% include [reusage-cli__feature-weight__desc__intro](../../_includes/work_src/reusage/cli__feature-weight__desc__intro.md) %}

### [first_feature_use_penalties](common.md#first_feature_use_penalties)

Command-line: `--first-feature-use-penalties`

{% include [reusage-cli__first-feature-use-penalties__intro](../../_includes/work_src/reusage/cli__first-feature-use-penalties__intro.md) %}

### [fixed_binary_splits](common.md#fixed_binary_splits)

Command-line: `--fixed-binary-splits`

A list of indices of binary features to put at the top of each tree.

### [penalties_coefficient](common.md#penalties_coefficient)

Command-line: `--penalties-coefficient`

A single-value common coefficient to multiply all penalties.

### [per_object_feature_penalties](common.md#per_object_feature_penalties)

Command-line: `--per-object-feature-penalties`

{% include [reusage-per-object-feature-penalties__intro](../../_includes/work_src/reusage/per-object-feature-penalties__intro.md) %}

### [model_shrink_rate](common.md#model_shrink_rate)

Command-line: `--model-shrink-rate`

The constant used to calculate the coefficient for multiplying the model on each iteration.

### [model_shrink_mode](common.md#model_shrink_mode)

Command-line: `model_shrink_mode`

Determines how the actual model shrinkage coefficient is calculated at each iteration.

## CTR settings

### [simple_ctr](ctr.md#simple_ctr)

{% include [reusage-cli__simple-ctr__intro](../../_includes/work_src/reusage/cli__simple-ctr__intro.md) %}

### [combinations_ctr](ctr.md#combinations_ctr)

{% include [reusage-cli__combination-ctr__intro](../../_includes/work_src/reusage/cli__combination-ctr__intro.md) %}

### [per_feature_ctr](ctr.md#per_feature_ctr)

{% include [reusage-cli__per-feature-ctr__intro](../../_includes/work_src/reusage/cli__per-feature-ctr__intro.md) %}

### [ctr_target_border_count](ctr.md#ctr_target_border_count)

{% include [reusage-cli__ctr-target-border-count__short-desc](../../_includes/work_src/reusage/cli__ctr-target-border-count__short-desc.md) %}

### [counter_calc_method](ctr.md#counter_calc_method)

The method for calculating the Counter CTR type.

### [max_ctr_complexity](ctr.md#max_ctr_complexity)

The maximum number of features that can be combined.

### [ctr_leaf_count_limit](ctr.md#ctr_leaf_count_limit)

The maximum number of leaves with categorical features. If the quantity exceeds the specified value a part of leaves is discarded.

### [store_all_simple_ctr](ctr.md#store_all_simple_ctr)

Ignore categorical features, which are not used in feature combinations, when choosing candidates for exclusion.

### [final_ctr_computation_mode](ctr.md#final_ctr_computation_mode)

Final CTR computation mode.

## Input file settings

### [-f, --learn-set](input.md#-f)

The path to the input file {% if audience == "internal" %} or table {% endif %} that contains the dataset description.

### [-t, --test-set](input.md#-t)

A comma-separated list of input files that contain the validation dataset description (the format must be the same as used in the training dataset).

### [--cd, --column-description](input.md#--cd)

The path to the input file {% if audience == "internal" %}or table{% endif %} that contains the [columns description](../../concepts/input-data_column-descfile.md).

### [--learn-pairs](input.md#--learn-pairs)

The path to the input file that contains the [pairs description](../../concepts/input-data_pairs-description.md) for the training dataset.

### [--test-pairs](input#--test-pairs)

The path to the input file that contains the [pairs description](../../concepts/input-data_pairs-description.md) for the validation dataset.

### [--learn-group-weights](input.md#--learn-group-weights)

The path to the input file that contains the weights of groups. Refer to the [Group weights](../../concepts/input-data_group-weights.md) section for format details.

### [--test-group-weights](input.md#--test-group-weights)

The path to the input file that contains the weights of groups for the validation dataset.

### [--learn-baseline](input.md#--learn-baseline)

The path to the input file that contains baseline values for the training dataset.

### [--test-baseline](input.md#--test-baseline)

The path to the input file that contains baseline values for the validation dataset.

### [--delimiter](input.md#--delimiter)

The delimiter character used to separate the data in the dataset description input file.

### [--has-header](input#--has-header)

Read the column names from the first line of the dataset description file if this parameter is set.

### [--params-files](input.md#--params-files)

The path to the input JSON file that contains the training parameters, for example:

### [--nan-mode](input.md#--nan-mode)

The method for [processing missing values](../../concepts/algorithm-missing-values-processing.md) in the input dataset.

## Multiclassification settings

### [classes_count](multiclassification.md#classes_count)

Command-line: `--classes-count`

{% include [reusage-classes-count__main-desc](../../_includes/work_src/reusage/classes-count__main-desc.md) %}

### [--class-names](multiclassification.md#--class-names)

This parameter is only for Command-line.

Classes names. Allows to redefine the default values when using the {{ error-function--MultiClass }} and {{ error-function--Logit }} metrics.

## Output settings

### [logging_level](output.md#logging_level)

Command line: `--logging-level`

The logging level to output to stdout.

### [metric_period](output.md#metric_period)

Command line: `--metric-period`

The frequency of iterations to calculate the values of [objectives and metrics](../../concepts/loss-functions.md).

The usage of this parameter speeds up the training.

### [verbose](output.md#verbose)

Command line: `--verbose`

_Alias:_`verbose_eval`

{% include [sections-with-methods-desc-python__feature-importances__verbose__short-description__list-intro](../../_includes/work_src/reusage/python__feature-importances__verbose__short-description__list-intro.md) %}

### [train_dir](output.md#train_dir)

Command line: `--train-dir`

The directory for storing the files generated during training.

### [model_size_reg](output.md#model_size_reg)

Command line: `--model-size-reg`

The model size regularization coefficient. The larger the value, the smaller the model size. Refer to the [Model size regularization coefficient](../../references/model-size-reg.md) section for details.

This regularization is needed only for models with categorical features (other models are small).

### [allow_writing_files](output.md#allow_writing_files)

Allow to write analytical and snapshot files during training.

### [save_snapshot](output.md#save_snapshot)

Enable snapshotting for [restoring the training progress after an interruption](../../features/snapshots.md).

### [snapshot_file](output.md#snapshot_file)

The name of the file to save the training progress information in. This file is used for [recovering training after an interruption](../../features/snapshots.md).

### [snapshot_interval](output.md#snapshot_interval)

The interval between saving snapshots in seconds.

### [roc_file](output.md#roc_file)

The name of the [output file](../../concepts/output-data_roc-curve-points.md) to save the ROC curve points to.

## Overfitting detection settings

### [early_stopping_rounds](overfitting-detection.md#early_stopping_rounds)

Sets the overfitting detector type to {{ fit--od-type-iter }} and stops the training after the specified number of iterations since the iteration with the optimal metric value.

### [od_type](overfitting-detection.md#od_type)

Command-line: `--od-type`

The type of the overfitting detector to use.

### [od_pval](overfitting-detection.md#od_pval)

Command-line: `--od-pval`

The threshold for the {{ fit--od-type-inctodec }} [overfitting detector](../../concepts/overfitting-detector.md) type.

### [od_wait](overfitting-detection.md#od_wait)

Command-line: `--od-wait`

The number of iterations to continue the training after the iteration with the optimal metric value.

## Performance settings

### [thread_count](performance.md#thread_count)

Command-line: `-T`, `--thread-count`

The number of threads to use during the training.

### [used_ram_limit](performance.md#used_ram_limit)

Command-line: `--used-ram-limit`

Attempt to limit the amount of used CPU RAM.

### [gpu_ram_part](performance.md#gpu_ram_part)

Command-line: `--gpu-ram-part`

How much of the GPU RAM to use for training.

### [pinned_memory_size](performance.md#pinned_memory_size)

Command-line: `--pinned-memory-size`

How much pinned (page-locked) CPU RAM to use per GPU.

### [gpu_cat_features_storage](performance.md#gpu_cat_features_storage)

Command-line: `--gpu-cat-features-storage`

The method for storing the categorical features' values.

### [data_partition](performance.md#data_partition)

Command-line: `--data-partition`

The method for splitting the input dataset between multiple workers.

## Processing unit settings

### [task_type](processing-unit.md#task_type)

Command line: `--task-type`

The processing unit type to use for training.

### [devices](#devices)

Command line: `--devices`

IDs of the GPU devices to use for training (indices are zero-based).

## Quantization settings

### [target_border](quantization.md#target_border)

Command-line: `--target-border`

If set, defines the border for converting target values to 0 and 1.

### [border_count](quantization.md#border_count)

Command-line: `-x`, `--border-count`

_Alias:_ `max_bin`

The number of splits for numerical features. Allowed values are integers from 1 to 65535 inclusively.

### [feature_border_type](quantization.md#feature_border_type)

Command-line: `--feature-border-type`

The [quantization mode](../../concepts/quantization.md) for numerical features.

### [per_float_feature_quantization](quantization.md#per_float_feature_quantization)

Command-line: `--per-float-feature-quantization`

The quantization description for the specified feature or list of features.

## Text processing parameters

These parameters are only for the Python package and Command-line version.

### [tokenizers](text-processing.md#tokenizers)

Command-line: `--tokenizers`

Tokenizers used to preprocess Text type feature columns before creating the dictionary.

### [dictionaries](text-processing.md#dictionaries)

Command-line: `--dictionaries`

{% include [reusage-cli__dictionaries__desc__div](../../_includes/work_src/reusage/cli__dictionaries__desc__div.md) %}

### [feature_calcers](text-processing.md#feature_calcers)

Command-line: `--feature-calcers`

{% include [reusage-cli__feature-calcers__desc__div](../../_includes/work_src/reusage/cli__feature-calcers__desc__div.md) %}

### [text_processing](text-processing.md#text_processing)

Command-line: `--text-processing`

{% include [reusage-cli__text-processing__div](../../_includes/work_src/reusage/cli__text-processing__div.md) %}

## Visualization settings

These parameters are only for the Python package.

### [name](visualization.md#name)

The experiment name to display in [visualization tools](../../features/visualization.md).
