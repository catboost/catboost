# GDBT speed benchmarks

This benchmark implements optimization algorithm on a pre-defined grid of hyper-parameters for every library. 
It measures time to perform boosting iteration and metric's value on test set using library primitives.
[XGBoost benchmark](https://xgboost.ai/2018/07/04/gpu-xgboost-update.html) code served as a starting point for this benchmark.

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
[Higgs](https://archive.ics.uci.edu/ml/datasets/HIGGS), [Epsilon](https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/), [MSRank](https://www.microsoft.com/en-us/research/project/mslr/), [CoverType](https://archive.ics.uci.edu/ml/datasets/covertype), [Airline](http://kt.ijs.si/elena_ikonomovska/data.html), [Synthetic](http://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_regression.html) [Abalone](https://archive.ics.uci.edu/ml/machine-learning-databases/abalone), [Letters](http://archive.ics.uci.edu/ml/datasets/Letter+Recognition).

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

Used library versions: CatBoost -- 0.10.3, XGBoost -- 0.80, LightGBM -- 2.2.2.
Hardware for GPU experiments: NVIDIA GTX1080TI.
Hardware for CPU experiments: Intel Xeon E312xx (Sandy Bridge) (32 cores).

Datasets present different task types (regression, classification, ranking), and have different sparsity number of features and samples.
The optimal parameters, for example maximal tree depth, will be different for each dataset.
Therefore we choose different hyper parameter grids for different datasets. The choice was made via manual running each algorithm with different parameters and watching at trends on learning curves.
Learning curves, distribution of time per iteration and other plots you may find [here](https://docs.google.com/spreadsheets/d/1_elljYjjdidKNbshY6vbMU5mwt4qDDHZVJxMFMg2q9M/edit?usp=sharing).

| Parameters grid                       |iterations| max_depth | learning_rate           | subsample       | reg_lambda  |
|---------------------------------------|----------|-----------|-------------------------|-----------------|-------------|
|Abalone                                |2000      |[6, 8, 10] |[.03, .07, .15]          |[.5, .75, 1]     |[1]          |
|Letters                                |2000      |[6, 8, 10] |[.03, .07, .15]          |[.5, .75, 1]     |[1]          |
|Epsilon                                |5000      |[4,6,8]    |[.09, .15, .25]          |[.5, 1]          |[1, 10, 100] |
|Higgs                                  |5000      |[6, 8, 10] |[.01, .03, .07, .15, .3] |[.25, .5, .75, 1]|[2, 4, 8, 16]|
|MSRank-RMSE                            |8000      |[6, 10, 12]|[.02, .05, .08]          |[.5, 1]          |[1, 10]      |
|MSRank-MultiClass                      |5000      |[6, 8, 10] |[.03, .07, .15]          |[.5, .75, 1]     |[1]          |
|Airlines                               |5000      |[6, 8, 10] |[.05]                    |default          |default      |
|"Synthetic (10M samples, 100 features)"|5000      |[6, 8, 10] |[.05, .15]               |default          |default      |
|"Synthetic (100K samples, 5K features)"|1000      |[6, 8]     |[.05]                    |default          |default      |

|Dataset                                | Metric| CatBoost | XGBoost | LightGBM |
|---------------------------------------|-------|----------|---------|----------|
|Abalone                                | RMSE  | 2.15400  | 2.15402 | 2.14308  |
|Letters                                | Error | 0.02024  | 0.02625 | 0.02275  |
|Epsilon                                | Error | 0.10868  | 0.11096 | 0.1112   |
|Higgs                                  | Error | 0.22713  | 0.22442 | 0.22238  |
|MSRank-RMSE                            | RMSE  | 0.73741  | 0.73837 | 0.73868  |
|MSRank-MultiClass                      | Error | 0.42405  | 0.4245  | 0.42405  |
|Airlines                               | Error | 0.18838  | 0.18954 | 0.17607  |
|"Synthetic (10M samples, 100 features)"| RMSE  | 8.36262  | 8.28738 | 9.35378  |
|"Synthetic (100K samples, 5K features)"| RMSE  | 6.8144   | 10.3923 | 11.0604  |

|Time per iteration, CPU                |CatBoost| XGBoost|LightGBM|
|---------------------------------------|--------|--------|--------|
|Abalone                                |0.008242|0.010857|0.003929|
|Letters                                |0.293273|0.161995|0.222528|
|Epsilon                                |0.270815|9.007035|1.340922|
|Higgs                                  |0.819541|1.331755|1.418482|
|MSRank-RMSE                            |0.159708|2.795780|0.519460|
|MSRank-MultiClass                      |0.646215|4.428504|1.523889|
|Airlines                               |7.652816|8.593439|9.099392|
|"Synthetic (10M samples, 100 features)"|0.659767|2.884988|2.279700|
|"Synthetic (100K samples, 5K features)"|0.251162|33.23975|2.071669|

|Time per iteration, GPU                |CatBoost| XGBoost|LightGBM|
|---------------------------------------|--------|--------|--------|
|Abalone                                |0.006382|0.00725 |0.02282 |
|Letters                                |0.019308|0.022959|0.10231 |
|Epsilon                                |0.0354  |2.6254  |0.7918  |
|Higgs                                  |0.06669 |0.157404|0.35668 |
|MSRank-RMSE                            |0.02227 |0.129005|0.48261 |
|MSRank-MultiClass                      |0.03144 |0.50291 |0.9992  |
|Airlines                               |0.625822|0.995749|3.56462 |
|"Synthetic (10M samples, 100 features)"|0.048836|0.251763|0.22827 |
|"Synthetic (100K samples, 5K features)"|0.05455 |1.54531 |4.3252  |

