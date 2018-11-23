#!/bin/bash -e

DATASETS_HOME=$HOME/datasets
BENCHMARKS_HOME=$HOME/benchmarks
TRAINING_OUTPUT_HOME=$HOME/training_output

DATASET=rossmann-store-sales

cd $DATASETS_HOME
mkdir -p $DATASET
cd $DATASET
#kaggle competitions download -c $DATASET

mkdir -p $TRAINING_OUTPUT_HOME/$DATASET

cd $BENCHMARKS_HOME/$DATASET

./preprocess_data.py

./catboost_early_stopping.py
./lightgbm_early_stopping.py
./xgboost_early_stopping.py

./catboost_experiment_sklearn_grid_cv.py
./lightgbm_experiment_sklearn_grid_cv.py
./xgboost_experiment_sklearn_grid_cv.py

./catboost_experiment_sklearn_random_cv.py
./lightgbm_experiment_sklearn_random_cv.py
./xgboost_experiment_sklearn_random_cv.py

./experiment_hyperopt.py cab
./experiment_hyperopt.py lgb
./experiment_hyperopt.py xgb
