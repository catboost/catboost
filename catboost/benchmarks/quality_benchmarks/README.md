# Benchmarks

## Comparison results

| | Default CatBoost | Tuned CatBoost | Default LightGBM | Tuned LightGBM | Default XGBoost | Tuned XGBoost | Default H2O | Tuned H2O
--- | --- | --- | --- | --- | --- | --- | --- | ---
[Adult](https://archive.ics.uci.edu/ml/datasets/Adult) | 0.272978 (±0.0004) (+1.20%) | _**0.269741**_ (±0.0001) | 0.287165 (±0.0000) (+6.46%) | 0.276018 (±0.0003) (+2.33%) | 0.280087 (±0.0000) (+3.84%) | 0.275423 (±0.0002) (+2.11%) | 0.276066 (±0.0000) (+2.35%) | 0.275104 (±0.0003) (+1.99%)
[Amazon](https://www.kaggle.com/c/amazon-employee-access-challenge) | 0.138114 (±0.0004) (+0.29%) | _**0.137720**_ (±0.0005) | 0.167159 (±0.0000) (+21.38%) | 0.163600 (±0.0002) (+18.79%) | 0.165365 (±0.0000) (+20.07%) | 0.163271 (±0.0001) (+18.55%) | 0.169497 (±0.0000) (+23.07%) | 0.162641 (±0.0001) (+18.09%)
[Appet](http://www.kdd.org/kdd-cup/view/kdd-cup-2009/Data) | _**0.071382**_ (±0.0002) (-0.18%) | 0.071511 (±0.0001) | 0.074823 (±0.0000) (+4.63%) | 0.071795 (±0.0001) (+0.40%) | 0.074659 (±0.0000) (+4.40%) | 0.071760 (±0.0000) (+0.35%) | 0.073554 (±0.0000) (+2.86%) | 0.072457 (±0.0002) (+1.32%)
[Click](http://www.kdd.org/kdd-cup/view/kdd-cup-2012-track-2) | 0.391116 (±0.0001) (+0.05%) | _**0.390902**_ (±0.0001) | 0.397491 (±0.0000) (+1.69%) | 0.396328 (±0.0001) (+1.39%) | 0.397638 (±0.0000) (+1.72%) | 0.396242 (±0.0000) (+1.37%) | 0.397853 (±0.0000) (+1.78%) | 0.397595 (±0.0001) (+1.71%)
[Internet](https://kdd.ics.uci.edu/databases/internet_usage/internet_usage.html) | 0.220206 (±0.0005) (+5.49%) | _**0.208748**_ (±0.0011) | 0.236269 (±0.0000) (+13.18%) | 0.223154 (±0.0005) (+6.90%) | 0.234678 (±0.0000) (+12.42%) | 0.225323 (±0.0002) (+7.94%) | 0.240228 (±0.0000) (+15.08%) | 0.222091 (±0.0005) (+6.39%)
[Kdd98](https://kdd.ics.uci.edu/databases/kddcup98/kddcup98.html) | 0.194794 (±0.0001) (+0.06%) | _**0.194668**_ (±0.0001) | 0.198369 (±0.0000) (+1.90%) | 0.195759 (±0.0001) (+0.56%) | 0.197949 (±0.0000) (+1.69%) | 0.195677 (±0.0000) (+0.52%) | 0.196075 (±0.0000) (+0.72%) | 0.195395 (±0.0000) (+0.37%)
[Kddchurn](http://www.kdd.org/kdd-cup/view/kdd-cup-2009/Data) | 0.231935 (±0.0004) (+0.28%) | _**0.231289**_ (±0.0002) | 0.235649 (±0.0000) (+1.88%) | 0.232049 (±0.0001) (+0.33%) | 0.233693 (±0.0000) (+1.04%) | 0.233123 (±0.0001) (+0.79%) | 0.232874 (±0.0000) (+0.68%) | 0.232752 (±0.0000) (+0.63%)
[Kick](https://www.kaggle.com/c/DontGetKicked) | 0.284912 (±0.0003) (+0.04%) | _**0.284793**_ (±0.0002) | 0.298774 (±0.0000) (+4.91%) | 0.295660 (±0.0000) (+3.82%) | 0.298161 (±0.0000) (+4.69%) | 0.294647 (±0.0000) (+3.46%) | 0.296355 (±0.0000) (+4.06%) | 0.294814 (±0.0003) (+3.52%)
[Upsel](http://www.kdd.org/kdd-cup/view/kdd-cup-2009/Data) | 0.166742 (±0.0002) (+0.37%) | _**0.166128**_ (±0.0002) | 0.171071 (±0.0000) (+2.98%) | 0.166818 (±0.0000) (+0.42%) | 0.168732 (±0.0000) (+1.57%) | 0.166322 (±0.0001) (+0.12%) | 0.169807 (±0.0000) (+2.21%) | 0.168241 (±0.0001) (+1.27%)


Metric: Logloss (lower is better). In the first brackets - std, in the second - the percentage difference from the tuned CatBoost.


You can find detailed information about experimental setup in [comparison description](https://github.com/catboost/benchmarks/blob/master/quality_benchmarks/comparison_description.pdf)

## Docker

* Docker has everything necessary for running experiments with Python.
* __Link:__  https://hub.docker.com/r/yandex/catboost_benchmarks/
* __To launch__:

```
docker run --workdir /root -v <path_to_local_folder>:/root/shared -p 80:8888 -it yandex/catboost_benchmarks sh -c "ipython notebook --ip=* --no-browser --allow-root"
```

## Files

* __experiment.py__:
   * The parent class that implements reading data, splitting data into subsets, calculating counter values, the cross-validation function, and the run function that starts the experiment.

* __\*\_experiment.py__:
   * Child classes that implement functions for converting data to the format of a specific algorithm, setting parameters to tune and the distributions they are selected from, as well as functions that start training.

* __cat_counter.py__:
   * Class for calculating counter values.

* __run.py__, __run_default.py__:
   * Files for starting experiments.

* __h2o\_preprocessing.py__:
   * Class for h2o data preprocessing.

* __run\_grid.R__:
   * File for starting h2o experiments in R (classification).

* __run\_grid\_reg.R__:
   * File for starting h2o experiments in R (regression).

* __run\_grid\_enc.R__:
   * File for starting h2o experiments in R (classification) using native h2o categorical_encoding options.

* __run\_grid\_reg\_enc.R__:
   * File for starting h2o experiments in R (regression) using native h2o categorical_encoding options.

* __run_install.R__:
   * File for installing necessary R libraries.

* __install/__:
   * The folder with the Docker file and scripts for installing the necessary libraries.

* __notebooks/__:
   * The folder containing notebooks with examples.

* __comparison_description.\*__:
   * PDF and TEX files describing the format of the comparison.

* __prepare\_\*/__
   * Scripts and notebooks for catboost train and test files preparation
## How to launch

* You can run it either from the command line or from the interpreter, importing the class you need.

* Parameters:
    ```
    Positional arguments:
      bst                            Algorithm name {cab, xgb, lgb}
      learning_task                  Type of task {classification, regression}

    Required arguments:
      --train                        Path to the train-file (str)
      --test                         Path to the test-file (str)
      --cd                           Path to the cd-file (str)

    Optional arguments:
      -t [ --n_estimators ]          Number of trees (int; by default 5000)
      -n [ --hyperopt_evals ]        Number of hyperopt iterations (int; by default 50)
      -o [ --output_folder_path ]    Path to the results folder (str; by default None
                                        and the result isn't stored)
      --holdout                      Size of the holdout section (float; by default 0)
      -h [ --help ]                  Help
   ```

* Usage:
    ```
    python run.py bst learning_task --train TRAIN_FILE_PATH --test TEST_FILE_PATH --cd CD_FILE_PATH
          [-t N_ESTIMATORS] [-n HYPEROPT_EVALS] [-o OUTPUT_FOLDER_PATH] [--holdout HOLDOUT] [-h]
    ```

* Launch examples
    * From the command line:
        ```
        python run.py cab classification -n 2 -t 10 -i --train amazon/train.txt
            --test amazon/test.txt --cd amazon/cd
        ```

    * From the interpreter:
        ```
        from catboost_experiment import CABExperiment
        cab_exp = CABExperiment('classification', n_estimators=10, max_hyperopt_evals=2,
            train_path='amazon/train.txt', test_path='amazon/test.txt', cd_path='amazon/cd')
        cab_exp.run()
        ```
## How to launch h2o experiments

* You can run it from the command line from the directory with train and test files. The results will be saved in the working directory. You can also run R and run the commands from here.

* H2O works smoother with preprocessed input (all categorical features are transformed to numerical), see the preprocessing script below. But it also works with raw data using its own categorical encoding methods (see `run_grid_enc.R` and `run_grid_reg_enc.R`).

* Files:
    ```
    Input:
      parsed_train                           Preprocessed train
      parsed_test                            Preprocessed test

    Output:
      result_default.tsv                     Results with default parameters
      result_tuned.tsv                       Results with tuned parameters
      result_default_seeds.tsv               Results with default parameters using a range of seeds
      result_tuned_seeds.tsv                 Results with tuned parameters using a range of seeds
   ```

* Usage:
    ```
    Rscript ./run_install.R
    Rscript ./run_grid.R
    ```

* Preprocessing:
The script creates two files in the `-o` directory using three input files.
```
python h2o_preprocessing.py <classification|regression> --train <path to train> --test <path to test> --cd <path to column desc> -o <output directory>
```

* Launch examples:
    ```
    > python h2o_preprocessing.py classification --train ./train_full3 --test ./test3 --cd ./train_full3.cd -o .
    > Rscript ../run_grid.R
    ...
    R is connected to the H2O cluster:
    H2O cluster version:        3.10.4.6
    H2O Connection ip:          localhost
    H2O Connection port:        23776
    R Version:                  R version 3.3.3 (2017-03-06)
    ...
    After starting H2O, you can use the Web UI at http://localhost:23776
    ```
