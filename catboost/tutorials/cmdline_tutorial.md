# Catboost command line tutorial
### Train and apply classification model

Train classification model with default params in silent mode.
Calc model predictions on custom data set, output will contain evaluated class1 probability:

```
catboost fit --learn-set train.tsv --test-set test.tsv --column-description train.cd  --loss-function Logloss
catboost calc -m model.bin --input-path custom\_data --cd train.cd -o custom\_data.eval -T 4 --prediction-type Probability
```

### Train regression model on csv file with header

Train classification model with 100 trees on comma separated pool with header:
```
catboost fit --learn-set train.csv --test-set test.csv --column-description train.cd  --loss-function RMSE --iterations 100 --delimiter=',' --has-header
```

### Train classification model in verbose mode with multiple error functions

It is possible to calc additional info while learning, such as current error on learn and current plus best error on test error. Remaining and passed time information is also displayed in verbose mode.
Custom loss functions parmeter allow to log additional error functions on learn and test for each iteration.
```
catboost fit --learn-set train --test-set test --column-description train.cd  --loss-function Logloss --custom-loss="AUC,Precision,Recall" -i 4 --verbose --print-trees
```
Example test\_error.tsv result:
```
iter    Logloss AUC     Precision       Recall
0       0.6638384193    0.8759125663    0.8537374221    0.9592193809
1       0.6350880554    0.8840660536    0.8565563873    0.9547779273
2       0.6098460477    0.8914710667    0.8609022556    0.9554508748
3       0.5834954183    0.8954216255    0.8608579414    0.9534320323
```

### Train classification with max used memory hint

Ctr computation on large pools can lead to out of memory problems, in such case it is possible to give catboost a hint about available memory size.
For example:
```
catboost fit --learn-set train.tsv --test-set test.tsv --column-description train.cd  --loss-function Logloss --used-ram-limit 4GB
```

### Random subspace method

To enable rsm for feature bagging use --rsm parameter:
```
catboost fit --learn-set train.tsv --test-set test.tsv --column-description train.cd  --loss-function Logloss --rsm 0.5
```

### Params file

For more convenience, you can create the params file:
```
{
    "thread_count": 4,
    "loss_function": "Logloss",
    "iterations": 400
}
```
And run the algorithm as follows:
```
catboost fit --learn-set train.tsv --test-set test.tsv --column-description train.cd --params-file params_file.txt
```
