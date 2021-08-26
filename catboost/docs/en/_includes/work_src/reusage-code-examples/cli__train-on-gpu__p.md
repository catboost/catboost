
Train a classification model on GPU:
```bash
catboost fit --learn-set ../pytest/data/adult/train_small --column-description ../pytest/data/adult/train.cd --task-type GPU
```
