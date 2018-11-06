# GDBT speed benchmarks

This benchmark implements optimization algorithm on a pre-defined grid of hyper-parameters for every library.
It measures time to perform boosting iteration and metric's value on test set using library primitives.
[XGBoost benchmark](https://xgboost.ai/2018/07/04/gpu-xgboost-update.html) code served as a starting point for this benchmark.

# Library dependencies

    xgboost==0.80
    matplotlib==2.2.2
    tqdm==4.26.0
    scipy==1.1.0
    lightgbm==2.2.1
    pandas==0.23.4
    catboost==0.10.3
    numpy==1.15.2
    scikit_learn==0.20.0

# Run benchmark

First of all you need to define a hyperparameters grid in json file, (you may find grid example in the root directory named example_params_grid.json)

Here every line define a hyperparameter name and a list of values to iterate in experiments.

    {
        "max_depth": [6, 10, 12],
        "learning_rate": [0.02, 0.05, 0.08],
        "reg_lambda": [1, 10],
        "subsample": [0.5, 1.0]
    }

This is example of how you can run an experiments on GPU for all three libraries for dataset airline using grid ''example_params_grid.json'' with 5000 iterations.

    python run.py --learners cat lgb xgb --datasets airline --params-grid example_params_grid.json --iterations 5000 --use-gpu

# Supported datasets
[Higgs](https://archive.ics.uci.edu/ml/datasets/HIGGS), [Epsilon](https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/), [MSRank (MSLR-WEB10K)](https://www.microsoft.com/en-us/research/project/mslr/), [CoverType](https://archive.ics.uci.edu/ml/datasets/covertype), [Airline](http://kt.ijs.si/elena_ikonomovska/data.html), [Synthetic](http://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_regression.html), [Abalone](https://archive.ics.uci.edu/ml/machine-learning-databases/abalone), [Letters](http://archive.ics.uci.edu/ml/datasets/Letter+Recognition).

Names for script run.py:

    abalone,airline,cover-type,epsilon,higgs,letters,msrank,msrank-classification,syntetic,synthetic-5k-features

# Draw plots:
It is supported to draw four types of plots:
1. Learning curves for top N runs between all experiments (parameter name -- *best*).
2. Box plot of time per iteration for each library (*time-per-iter*).
3. Draw time to achieve percent from best quality (*quality-vs-time*).
4. Learning curves for concrete experiments (*custom*).

Third method calculates the best achieved quality between all experiments. Then fix a grid of relative percents of this value. Then for each level of quality the method filter all experiments from grid search that reach it and compute median (circle point), minimum and maximum time when algorithm achieved that score.

#### Examples
Draw learning curves for 5 best experiments on dataset MSRank (multiclass mode):

    python plot.py --type best --top 5 --from-iter 500 -i ./results/MSRank-MultiClass/ -o msrank_mc_plots

Draw time curves with figure size 10x8 for experiments on Higgs dataset starting from 85% of best achieved quality to 100% in 30 points:

    python plot.py --type quality-vs-time -f 10 8 -o higgs_plots -i ./results/Higgs/ --low-percent 0.85 --num-bins 30

You can pass option *--only-min* in this mode, to draw only minimum time to achive percent of quality.

Draw time per iteration box plots for every library:

    python plot.py --type time-per-iter -f 10 8 -o higgs_plots -i ./results/Higgs

If you want to draw experiments of concrete method pass option *--only method_name*. Choices for method_name are cat, xgb, lgb, cpu, gpu and combinations of those.

# Experiments

Used library versions: CatBoost -- 0.10.3, XGBoost -- 0.80, LightGBM -- 2.2.1.
Hardware for GPU experiments: NVIDIA GTX1080TI.
Hardware for CPU experiments: Intel Xeon E312xx (Sandy Bridge) (32 cores).

We took the most popular datasets between benchmarks in other libraries and kaggle competitions.
Datasets present different task types (regression, classification, ranking), and have different sparsity number of features and samples.
This set of datasets cover the majority of computation bottlenecks in gbdt algorithm, for example with large number of samples computation of loss function gradients may become bottleneck, with large number of features the phase of searching of the tree structure may become computational bottleneck. To demonstrate performance of each library in such experiments we created two synthetic datasets -- the first with 10 million objects and 100 features, the second with 100K objects and 5K features.

In such different tasks the optimal hyper parameters, for example maximal tree depth or learning rate, will be different.
Thus we choose different hyper parameter grids for different datasets.
The selection of the grid for optimization consisted of the following steps:

1. Run each library with different maximum depth (or number of leaves in case of LightGBM) and learning rate.
2. Select best tries for each library and run with different sample rate and regularization parameters.
3. Create grid based on all observations. It must contain all best tries for every algorithm.

Learning curves, distribution of time per iteration and other plots you may find in [plots](./plots) folder.

Table with dataset characteristics. Sparsity values was calculated using the formula `sparsity = (100 - 100 * float(not_null_count) / (pool.shape[0] * pool.shape[1]))`, where `not_null_count` is number of NaN, None or zero (abs(x) < 1e-6) values in dataset.

|Dataset characteristics              |\#samples   |\#features|sparsity|
|-------------------------------------|------------|----------|--------|
|Abalone                              | 4.177      | 8        | 3.923  |
|Letters                              | 20.000     | 16       | 2.621  |
|Epsilon                              | 500.000    | 2000     | 3.399  |
|Higgs                                | 11.000.000 | 28       | 7.912  |
|MSRank                               | 1.200.192  | 137      | 37.951 |
|Airlines                             | 115.069.017| 13       | 7.821  |
|Synthetic (10M samples, 100 features)| 10.000.000 | 100      | 0.089  |
|Synthetic (100K samples, 5K features)| 100.000    | 5000     | 0.079  |

| Parameters grid                     |iterations| max_depth | learning_rate           | subsample       | reg_lambda  |
|-------------------------------------|----------|-----------|-------------------------|-----------------|-------------|
|Abalone                              |2000      |[6, 8, 10] |[.03, .07, .15]          |[.5, .75, 1]     |[1]          |
|Letters                              |2000      |[6, 8, 10] |[.03, .07, .15]          |[.5, .75, 1]     |[1]          |
|Epsilon                              |5000      |[4,6,8]    |[.09, .15, .25]          |[.5, 1]          |[1, 10, 100] |
|Higgs                                |5000      |[6, 8, 10] |[.01, .03, .07, .15, .3] |[.25, .5, .75, 1]|[2, 4, 8, 16]|
|MSRank-RMSE                          |8000      |[6, 10, 12]|[.02, .05, .08]          |[.5, 1]          |[1, 10]      |
|MSRank-MultiClass                    |5000      |[6, 8, 10] |[.03, .07, .15]          |[.5, .75, 1]     |[1]          |
|Airlines                             |5000      |[6, 8, 10] |[.05]                    |default          |default      |
|Synthetic (10M samples, 100 features)|5000      |[6, 8, 10] |[.05, .15]               |default          |default      |
|Synthetic (100K samples, 5K features)|1000      |[6, 8]     |[.05]                    |default          |default      |

Table of quality values. Metric abbreviations: RMSE -- root mean squared error. Error -- $ # incorrect classified samples / # samples $.

|Dataset                              | Metric|CatBoost|XGBoost|LightGBM|
|-------------------------------------|-------|--------|-------|--------|
|Abalone                              | RMSE  | 2.154  | 2.154 | 2.143  |
|Letters                              | Error | 0.020  | 0.026 | 0.023  |
|Epsilon                              | Error | 0.109  | 0.111 | 0.111  |
|Higgs                                | Error | 0.227  | 0.224 | 0.222  |
|MSRank-RMSE                          | RMSE  | 0.737  | 0.738 | 0.739  |
|MSRank-MultiClass                    | Error | 0.424  | 0.425 | 0.424  |
|Airlines                             | Error | 0.188  | 0.190 | 0.176  |
|Synthetic (10M samples, 100 features)| RMSE  | 8.363  | 8.287 | 9.354  |
|Synthetic (100K samples, 5K features)| RMSE  | 6.814  | 10.39 | 11.060 |

This table present median time spending algorithm on boosting iteration over all experiments in grid measured in seconds.

|Time per iteration, seconds, CPU     |CatBoost|XGBoost|LightGBM|
|-------------------------------------|-----|------|-----|
|Abalone                              |0.008|0.011 |0.004|
|Letters                              |0.293|0.162 |0.223|
|Epsilon                              |0.271|9.007 |1.341|
|Higgs                                |0.820|1.332 |1.419|
|MSRank-RMSE                          |0.160|2.796 |0.520|
|MSRank-MultiClass                    |0.646|4.428 |1.524|
|Airlines                             |7.653|8.593 |9.099|
|Synthetic (10M samples, 100 features)|0.660|2.885 |2.280|
|Synthetic (100K samples, 5K features)|0.251|33.240|2.072|

|Time per iteration, seconds, GPU     |CatBoost|XGBoost|LightGBM|
|-------------------------------------|-----|-----|-----|
|Abalone                              |0.006|0.007|0.023|
|Letters                              |0.019|0.023|0.102|
|Epsilon                              |0.035|2.625|0.792|
|Higgs                                |0.067|0.157|0.357|
|MSRank-RMSE                          |0.022|0.129|0.483|
|MSRank-MultiClass                    |0.031|0.503|0.999|
|Airlines                             |0.626|0.996|3.565|
|Synthetic (10M samples, 100 features)|0.049|0.252|0.228|
|Synthetic (100K samples, 5K features)|0.055|1.545|4.325|
