# Cross-validation

## {{ dl--purpose }} {#purpose}

{% include [reusage-cli__cross-validation__purpose__div](../_includes/work_src/reusage/cli__cross-validation__purpose__div.md) %}


## {{ dl__cli__execution-format }} {#execution-format}

```
catboost fit -f <file path> --cv <cv_type>:<fold_index>;<fold_count> [--cv-rand <value>] [other parameters]
```

For example:
```
catboost fit -f train.tsv --cv Classical:0;5
```

## {{ common-text__title__reference__parameters }} {#options}

{% include [reusage-cli-cross-validation__options-desc](../_includes/work_src/reusage/cli-cross-validation__options-desc.md) %}


## {{ dl__usage-examples }} {#usage-examples}

Launch the training three times with the same partition random seed and different validation folds to run a three-fold cross-validation:
```
catboost fit -f train.tsv --cv Classical:0;3 --cv-rand 17 --test-err-log fold_0_error.tsv
catboost fit -f train.tsv --cv Classical:1;3 --cv-rand 17 --test-err-log fold_1_error.tsv
catboost fit -f train.tsv --cv Classical:2;3 --cv-rand 17 --test-err-log fold_2_error.tsv
```

These trainings generate files with metric values, which can be aggregated manually.

