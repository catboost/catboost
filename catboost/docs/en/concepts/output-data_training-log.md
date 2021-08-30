# Metrics and time information

#### {{ output--contains }}

- {% include [format-common-phrases-format__contains__metrics](../_includes/work_src/reusage-formats/format__contains__metrics.md) %}

- Information about the number of seconds of training:
    - passed since the beginning
    - remaining until the end


#### {{ output--format }}

The resulting JSON file consists of the following arrays:

- [meta](#meta__array)
- [iterations](#iterations__array)

#### meta

Contains basic information about the training.

{% include [format-common-phrases-json__format__array](../_includes/work_src/reusage-formats/json__format__array.md) %}


```
"meta": {
    "`launch_mode`": "Train",
    "`name`": "second",
    "`iteration_count`": 1000,
    "`learn_metrics`": [
      {
        "`name`": "Precision:class=0",
        "`value`": "Max"
      },
      {
        "`name`": "Precision:class=1",
        "`value`": "Max"
      }
    ],
    "`test_sets`": [
      "test",
      ...
      "testN"
    ],
    "`test_metrics`": [
      {
        "`name`":"Precision:class=0",
        "`value`":"Max"
      },
      {
        "`name`":"Precision:class=1",
        "`value`":"Max"
      }
    ],
    "`learn_sets`": [
      "learn"
    ]
}
```


Property | Type | Description
----- | ----- | -----
`launch_mode` | {{ json__string }} | The specified launch mode.<br/><br/>Possible values:<br/>- Train — Training launch mode.<br/>- CV — Cross-validation launch mode (for the Python [cv](python-reference_cv.md) method only).<br/><br/>The command-line implementation of the [Cross-validation](cli-reference_cross-validation.md) feature returns the Train value in this parameter.<br/><br/>{% endnote %}
`name` | {{ json__string }} | The experiment name.<br/><br/>The value can be set in the `--name` (`--name`) training parameter. The default name is .
`iteration_count` | {{ json__int }} | The maximum number of trees that can be built when solving machine learning problems.<br>The final number of iterations may be less than the output in this property.
`learn_metrics` | {{ json__array }} | A list of metrics calculated for the learning dataset and information regarding the optimization method.
`test_sets` | {{ json__array }} | The names of the arrays within the [iterations](#iterations__array) array that contain the calculated values of metrics for the validation datasets.
`test_metrics` | {{ json__array }} | A list of metrics calculated for the validation dataset and information regarding the optimization method.
`name` | {{ json__string }} | The name of the metric.
`value` | {{ json__string }} | The method for defining the best value of the metric. Possible values:<br/>- Min — The smaller the value of the metric, the better.<br/>- Max — The bigger the value of the metric, the better.<br/>- Undefined — The best value of the metric is not defined.<br/>- Float value — The best value of the metric is user-defined.
`learn_sets` | {{ json__array }} | The name of the array within the [iterations](#iterations__array) array that contains the calculated values of the metrics for the learning dataset.


#### iterations

Contains an array of [metric](../concepts/loss-functions.md) values for the training and test sets and information on the duration of training for each iteration.

{% include [format-common-phrases-json__format__array](../_includes/work_src/reusage-formats/json__format__array.md) %}


```
"iterations": [
    {
      "`learn`": [
        0.8333333333,
        0.6666666667,
        0.7325581395,
        -1.0836257,
        0.4347826087,
        0.1428571429,
        0.984375,
        -0.6881395691
      ],
      "`iteration`": 0,
      "`passed_time`": 0.0227411829,
      "`remaining_time`": 22.71844172,
      "`test`1": [
        0.8333333333,
        0.6666666667,
        0.7325581395,
        -1.0836257,
        0.4347826087,
        0.1428571429,
        0.984375,
        -0.6881395691
      ],
      ...
      "`test`N": [
        0.7333453333,
        0.3666664267,
        0.0325581395,
        -1.9046257,
        0.8937826089,
        0.4138571478,
        0.004313,
        -0.3881390984
      ]
    }
  ]
```

Property | Type | Description
----- | ----- | -----
`learn` | {{ json__array }} | A list of metric values calculated for the learning dataset. The order of metrics is given in the `learn_metrics` array of the [meta](#meta__array) array.
`iteration` | {{ json__int }} | The index of the iteration. Numbering starts from zero.
`passed_time` | {{ json__float }} | The number of seconds passed since the beginning of training.
`remaining_time` | {{ json__float }} | The number of seconds remaining until the end of training given that all the scheduled iterations take place.
`test` | {{ json__array }} | The values of metrics calculated for the corresponding validation dataset.<br/><br/>The order of the metrics is given in the `test_metrics` array of the [meta](#meta__array) array.


#### {{ output--example }}

```
{
  "meta": {
    "`launch_mode`": "Train",
    "`name`": "second",
    "`iteration_count`": 1000,
    "`learn_metrics`": [
      {
        "`name`": "Precision:class=0",
        "`value`": "Max"
      },
      {
        "`name`": "Precision:class=1",
        "`value`": "Max"
      },
      {
        "`name`": "Precision:class=2",
        "`value`": "Max"
      },
      {
        "`name`": "MultiClass",
        "`value`": "Max"
      },
      {
        "`name`": "Recall:class=0",
        "`value`": "Max"
      },
      {
        "`name`": "Recall:class=1",
        "`value`": "Max"
      },
      {
        "`name`": "Recall:class=2",
        "`value`": "Max"
      },
      {
        "`name`": "MultiClassOneVsAll",
        "`value`": "Max"
      }
    ],
    "`test_sets`": [
      "test"
    ],
    "`test_metrics`": [      
      {
        "`name`": "Precision:class=0",
        "`value`": "Max"
      },
      {
        "`name`": "Precision:class=1",
        "`value`": "Max"
      },
      {
        "`name`": "Precision:class=2",
        "`value`": "Max"
      },
      {
        "`name`": "MultiClass",
        "`value`": "Max"
      },
      {
        "`name`": "Recall:class=0",
        "`value`": "Max"
      },
      {
        "`name`": "Recall:class=1",
        "`value`": "Max"
      },
      {
        "`name`": "Recall:class=2",
        "`value`": "Max"
      },
      {
        "`name`": "MultiClassOneVsAll",
        "`value`": "Max"
      }
    ],
    "`learn_sets`": [
      "learn"
    ]
  },
  "iterations": [
       {
       "`learn`": [
       0.8333333333,
       0.6666666667,
       0.7325581395,
       -1.0836257,
       0.4347826087,
       0.1428571429,
       0.984375,
       -0.6881395691
       ],
       "`iteration`": 0,
       "`passed_time`": 0.0227411829,
       "`remaining_time`": 22.71844172,
       "`test`": [
       0.8333333333,
       0.6666666667,
       0.7325581395,
       -1.0836257,
       0.4347826087,
       0.1428571429,
       0.984375,
       -0.6881395691
      ]
    },
    {
      "`learn`": [
        0.7142857143,
        1,
        0.7820512821,
        -1.068965402,
        0.652173913,
        0.1428571429,
        0.953125,
        -0.6832264
      ],
      "`iteration`": 1,
      "`passed_time`": 0.04471753966,
      "`remaining_time`": 22.31405229,
      "`test`": [
        0.7142857143,
        1,
        0.7820512821,
        -1.068965402,
        0.652173913,
        0.1428571429,
        0.953125,
        -0.6832264
      ]
    },
    ...
  ]
}
```
