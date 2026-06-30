# Graph aggregated features 

If graph information is provided, the dataset will be augmented with additional features.

This is an implementation of the approach described in the [TabGraphs: A Benchmark and Strong Baselines for Learning on Graphs with Tabular Node Features](https://arxiv.org/abs/2409.14500) paper.

For each float feature $i$, the sample $F(v)$ will be augmented with the following aggregated features:
- $mean \; { F_i(u) | (v, u) \in Graph}$,  
- $max \; { F_i(u) | (v, u) \in Graph}$,
- $min \; { F_i(u) | (v, u) \in Graph}$,

where $F_i(u)$ is a feature $i$ of the sample $u$.

{% note info %}

Feature aggregated from ignored feature will be also ignored.

If graph was used for model training, graph information will be also required for all action on model with dataset (i.e. applying, fstr calculation and so on).

{% endnote %}
