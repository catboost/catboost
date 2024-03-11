# Train a model

{% note info %}

Training on GPU requires NVIDIA Driver of version 450.80.02 or higher.

{% endnote %}

## Execution format {#execution-format}

```text
catboost fit -f <file path> [optional parameters]
```

## Options {#options}

See all options in the [Training parameters](../references/training-parameters/index.md) section.

## Usage examples {#usage-examples}

Train a model with 100 trees on a comma-separated pool with header:

```text
catboost fit --learn-set train.csv --test-set test.csv --column-description train.cd  --loss-function RMSE --iterations 100 --delimiter=',' --has-header
```

Train a classification model on GPU:

```text
catboost fit --learn-set ../pytest/data/adult/train_small --column-description ../pytest/data/adult/train.cd --task-type GPU
```
