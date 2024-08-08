# Graph description

#### {{ input_data__title__contains }}
A description of graph structure in the edge list format. 

Graph information is used to calculate the [graph aggregated features](../features/graph-aggregated-features.md).

{% note info %}

Only one thing can be passed, either a graph or [pairs description](../concepts/input-data_pairs-description.md).

Can be used only when [dataset description](../concepts/input-data_values-file.md) contains the GroupId column. 

If graph was used for model training, graph information will be also required for all action on model with dataset (applying, fstr calculation and so on).

{% endnote %}

#### {{ input_data__title__specification }}

- List each edge description on a new line.
- Use a tab as the delimiter to separate the columns on a line.

#### {{ input_data__title__row-format }}

```
<start_vertex index><\t><end_vertex index>
```

- `start_vertex index`is the zero-based index of the start_vertex object from the graph.
- `end_vertex index`is the zero-based index of the loser object from the graph.

#### {{ input_data__title__example }}

```
2<\t>1
2<\t>0
7<\t>6
```

