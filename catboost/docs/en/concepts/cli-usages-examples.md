# Usage examples

## Train and apply a classification model {#class-model}

Train a classification model with default parameters in silent mode and then calculate model predictions on a custom dataset. The output contains the evaluated class1 probability:

```no-highlight
catboost fit --learn-set train.tsv --test-set test.tsv --column-description train.cd  --loss-function Logloss

catboost calc -m model.bin --input-path custom_data --cd train.cd -o custom_data.eval -T 4 --prediction-type Probability
```


## Train a regression model on a CSV file with header {#regression-csv}

Train a model with 100 trees on a comma-separated pool with header:

```no-highlight
catboost fit --learn-set train.csv --test-set test.csv --column-description train.cd  --loss-function RMSE --iterations 100 --delimiter=',' --has-header
```


## Train a classification model in verbose mode with multiple error functions  {#verbose-custom-loss}

The <q>Verbose</q> logging level mode allows to output additional calculations while learning, such as current learn error or current plus best error on test error. Remaining and elapsed time are also displayed.

The `--custom-metric` parameter allows to log additional error functions on learn and test for each iteration.

```no-highlight
catboost fit --learn-set train --test-set test --column-description train.cd  --loss-function Logloss --custom-loss="AUC,Precision,Recall" -i 4 --logging-level Verbose
```

Example `test_error.tsv` result:

```no-highlight
iter    Logloss         AUC             Precision       Recall
0       0.6638384193    0.8759125663    0.8537374221    0.9592193809
1       0.6350880554    0.8840660536    0.8565563873    0.9547779273
2       0.6098460477    0.8914710667    0.8609022556    0.9554508748
3       0.5834954183    0.8954216255    0.8608579414    0.9534320323
```


## Train a classification model with a preferred memory limit {#used-ram-limit}

Ctr computation on large pools can lead to <q>out of memory</q> problems. In this case it is possible to give Catboost a hint about available memory:
```no-highlight
catboost fit --learn-set train.tsv --test-set test.tsv --column-description train.cd  --loss-function Logloss --used-ram-limit 4GB
```


## Train a model on GPU {#cli__train-on-gpu}

Train a classification model on GPU:
```bash
catboost fit --learn-set ../pytest/data/adult/train_small --column-description ../pytest/data/adult/train.cd --task-type GPU
```


## Random subspace method  {#rsm}

To enable random subspace method for feature bagging use the `--rsm` parameter:
```no-highlight
catboost fit --learn-set train.tsv --test-set test.tsv --column-description train.cd  --loss-function Logloss --rsm 0.5
```


## Calculate the object importances {#ostr}

To calculate the object importances:
1. Train the model:

    ```no-highlight
    catboost fit --loss-function Logloss -f train.tsv -t test.tsv --column-description train.cd
    ```

1. Calculate the object importances using the trained model:
    ```no-highlight
    catboost ostr -f train.tsv -t test.tsv --column-description train.cd -o object_importances.tsv
    ```

