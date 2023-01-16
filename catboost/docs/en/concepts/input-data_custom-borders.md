# Custom quantization borders and missing value modes

#### {{ output--contains }}

Custom quantization borders and the method for processing missing values for the dataset numerical features.

#### {{ output--format }}

- Each line contains information regarding a single border and optionally the missing values mode settings for the corresponding feature.
- Different missing values mode settings can not be specified for a single feature. The value either has to be set only on one line or should be the same on different lines that contain information regarding a single feature.
- Supported missing values modes:

  {% include [reusage-missing-values-mv-processing-methods](../_includes/work_src/reusage-missing-values/mv-processing-methods.md) %}

- The missing values mode for the feature is defined by the value of the `--nan-mode` (`nan_mode`) training parameter if not specified in this file.
- Format of a single line:
    ```
    <zero-based feature ID><\t><border value><\t><missing values mode (optional)>
    ```


#### {{ output--example }}
The following description contains two borders for features indexed 0 and 2 and missing values settings for each of these features:
```
0<\t>0.25<\t>{{ fit__nan_mode__forbidden }}
0<\t>0.75<\t>{{ fit__nan_mode__forbidden }}
2<\t>0.3
2<\t>0.85<\t>{{ fit__nan_mode__max }}
```

