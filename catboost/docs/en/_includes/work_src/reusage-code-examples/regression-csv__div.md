
Train a model with 100 trees on a comma-separated pool with header:

```no-highlight
catboost fit --learn-set train.csv --test-set test.csv --column-description train.cd  --loss-function RMSE --iterations 100 --delimiter=',' --has-header
```
