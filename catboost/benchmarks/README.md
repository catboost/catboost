# Benchmarks

This repo contains different CatBoost benchmarks.

## Quality: comparison with other libraries
Go to subdirectory [quality benchmarks](https://github.com/catboost/benchmarks/blob/master/quality_benchmarks/) to see quality benchmarks.
These are benchmarks in binary classification mode. They compare CatBoost vs XGBoost vs LightGBM vs H20.

## Training speed: comparisons with other libraries

You can find scripts to run LigthGBM/XGBoost/CatBoost CPU and GPU versions and compare its runtime in [training speed](https://github.com/catboost/benchmarks/blob/master/training_speed/) subdirectory

## Training speed: CPU vs GPU

This benchmark shows speedup of GPU over CPU on different dataset sizes and on different devices.

## Applier speed: comparison with other libraries

Benchmarks with comparison of applier speed with other libraries are in folder [model evaluation speed](https://github.com/catboost/benchmarks/blob/master/model_evaluation_speed/)

## Ranking: compare quality of different GBDT libraries and different modes

This benchmark shows how different libraries and modes perform on existing open source ranking datasets.

## SHAP values calculation speed: comparison with others
 
Shap values calculation benchmarks are in [shap speed](./shap_speed/) subdirectory.
This benchmark will show the complexity of SHAP calculation for each library. And will show a speed comparison on a fixed dataset.

## Kaggle

This is the folder where we are adding quality comparisons on some kaggle datasets.
Currently, it only contains comparison of different libraries on Rossman store sales competition.
