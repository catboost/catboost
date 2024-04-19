# Custom quantization borders and missing value modes

#### {{ output--contains }}

{% include [custom-borders-custom-borders__contains__intro](../_includes/work_src/reusage-formats/custom-borders__contains__intro.md) %}


#### {{ output--format }}

- Each line contains information regarding a single border and optionally the missing values mode settings for the corresponding feature.
- Missing value modes are output for a feature if the following conditions are met at the same time:
    - The chosen missing value mode for the feature differs from {{ fit__nan_mode__forbidden }}.
    - Missing values are present in the dataset.

    The global missing value mode is specified in the `--nan-mode` (`nan_mode`) training parameter and can be overridden in the [Custom quantization borders and missing value modes](input-data_custom-borders.md) input file.

- Format of a single line:
    ```
    <zero-based feature ID><\t><border value><\t><missing value mode>
    ```


#### {{ output--example }}

```
0<\t>0.25
0<\t>0.75
2<\t>0.3<\t>{{ fit__nan_mode__max }}
2<\t>0.85<\t>{{ fit__nan_mode__max }}
```

