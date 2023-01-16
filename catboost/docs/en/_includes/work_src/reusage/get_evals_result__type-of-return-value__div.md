
{{ python-type--dict }}

Output format:
```
{pool_name: {metric_name_1-1: [value_1, value_2, .., value_N]}, .., {metric_name_1-M: [value_1, value_2, .., value_N]}}
```

For example:
```
{'learn': {'Logloss': [0.6720840012056274, 0.6476800666988386, 0.6284055381249782], 'AUC': [1.0, 1.0, 1.0], 'CrossEntropy': [0.6720840012056274, 0.6476800666988386, 0.6284055381249782]}}
```
