# Known limitations

* Windows is not supported. Work in progress.
* GPU is not supported. Work in progress.
* Text and embeddings features are not supported. Work in progress.
* Feature distribution statistics (like `calc_feature_statistics`on CatBoost python package) with datasets on Spark is not supported. But it is possible to run such analysis with models exported to local files in usual CatBoost format.
* Generic string class labels are not supported. String class labels can be used only if these strings represent integer indices.
* ``boosting_type=Ordered`` is not supported.
* Training of models with non-symmetric trees is not supported. But such models can be loaded and applied on datasets in Spark.
* Monotone constraints are not supported.
* Multitarget training is not supported.
* Stochastic Gradient Langevin Boosting mode is not supported.
