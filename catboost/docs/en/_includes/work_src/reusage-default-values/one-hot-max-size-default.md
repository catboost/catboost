The default value depends on various conditions:

- N/A if training is performed on CPU in Pairwise scoring mode

    {% cut "Read more about  Pairwise scoring" %}


    {% include [reusage-default-values-metrics_parwise_scoring](metrics_parwise_scoring.md) %}


    {% endcut %}

- 255 if training is performed on GPU and the selected Ctr types require target data that is not available during the training
- 10 if training is performed in [Ranking](../../../concepts/loss-functions-ranking.md) mode
- 2 if none of the conditions above is met
