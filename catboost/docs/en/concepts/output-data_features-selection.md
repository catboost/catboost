# Features selection result

#### {{ output--contains }}
The results of features selection.
#### {{ output--format }}

JSON with results of features selection. Contains the following fields:

- `selected_features` — the array with indices of selected features.
- `selected_features_names` — the array with names of selected features (if feature names are present in column description file).
- `eliminated_features` — the array with indices of eliminated features.
- `eliminated_features_names` — the array with names of eliminated features (if feature names are present in column description file).
- `loss_graph` — the points for building a plot of loss values.
    - `removed_features_count` — the number of removed features at each point.
    - `loss_values` — the corresponding loss values at each point.
    - `main_indices` — indices of main points on the plot (where model was really trained).
    

#### {{ output--example }}

```json
{
    "eliminated_features":
      [
        4,
        8,
        5,
        1,
        2
      ],
    "eliminated_features_names":
      [
        "C3",
        "C5",
        "F1",
        "C1",
        "C2"
      ],
    "loss_graph":
      {
        "loss_values":
          [
            0.3394293155,
            0.3354862546,
            0.3347135661,
            0.3342145235,
            0.3342145235,
            0.3342145235
          ],
        "main_indices":
          [
            0,
            5
          ],
        "removed_features_count":
          [
            0,
            1,
            2,
            3,
            4,
            5
          ]
        },
    "selected_features":
      [
        3,
        6,
        7
      ],
    "selected_features_names":
      [
        "F0",
        "C4",
        "F2"
      ]
    }
```

