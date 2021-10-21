# Spark cluster configuration


{{ catboost-spark }} requires one training task per executor. If you run training, you have to set
`spark.task.cpus` parameter to be equal to the number of cores in executors (`spark.executor.cores`).
This limitation might be relaxed in the future ([the corresponding issue #1622](https://github.com/catboost/catboost/issues/1622)).

Model application or feature importance evaluation do not have this limitation.
