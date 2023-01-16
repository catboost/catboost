# Cli reusage

Possible values:

- {{ fit__model-format_CatboostBinary }}.
- {{ fit__model-format_applecoreml }}(only datasets without categorical features are currently supported).
- {{ fit__model-format_cpp }} (multiclassification models are not currently supported). See the [C++](../../../concepts/c-plus-plus-api_applycatboostmodel.md) section for details on applying the resulting model.
- {{ fit__model-format_python }} (multiclassification models are not currently supported).See the [Python](../../../concepts/python-reference_apply_catboost_model.md) section for details on applying the resulting model.
- {{ fit__model-format_json }} (multiclassification models are not currently supported). Refer to the [CatBoost JSON model tutorial]({{ catboost-repo__json-tutorial }}) for format details.
- {{ fitpython__model-format_onnx }} — ONNX-ML format (only datasets without categorical features are currently supported). Refer to [https://onnx.ai/](https://onnx.ai/) for details. See the [ONNX](../../../concepts/apply-onnx-ml.md) section for details on applying the resulting model.
- {{ fitpython__model-format_pmml }} — [PMML version {{ pmml-supported-version }}]({{ pmml-v4point3 }}) format. Categorical features must be interpreted as one-hot encoded during the training if present in the training dataset. This can be accomplished by setting the `--one-hot-max-size`/`one_hot_max_size` parameter to a value that is greater than the maximum number of unique categorical feature values among all categorical features in the dataset. See the [PMML](../../../concepts/apply-pmml.md) section for details on applying the resulting model.


An up-to-date list of available {{ product }} releases and the corresponding binaries for different operating systems is available in the **Download** section of the [releases]({{ releases-page }}) page on GitHub.

Apply the model.

Train the model.

- `{{ cd-file__col-type__label }}`
- `{{ cd-file__col-type__Baseline }}`
- `{{ cd-file__col-type__Weight }}`
- `{{ cd-file__col-type__SampleId }}` (`{{ cd-file__col-type__DocId }}`)
- `{{ cd-file__col-type__GroupId }}` (`{{ cd-file__col-type__QueryId }}`)
- `{{ cd-file__col-type__QueryId }}`
- `{{ cd-file__col-type__SubgroupId }}`
- `{{ cd-file__col-type__Timestamp }}`
- `{{ cd-file__col-type__GroupWeight }}`


## Penalties format {#penalties-format}

Supported formats for setting the value of this parameter:

- Set the penalty for each feature as a string (the number of features is n).

    {% cut "Format" %}

    ```
    "(<feature-penalty_0>, <feature-penalty_2>, .., <feature-penalty_n-1>)"
    ```

    {% note info %}

    Spaces between values are not allowed.

    {% endnote %}

    {% endcut %}

    Values should be passed as a parenthesized string of comma-separated values. Penalties equal to 0 at the end of the list may be dropped.

    In this
    {% cut "example" %}

    `--first-feature-use-penalties` parameter:

    ```
    --first-feature-use-penalties "(0.1,1,3)"
    ```

    `--per-object-feature-penalties` parameter:

    ```
    --per-object-feature-penalties "(0.1,1,3)"
    ```

    {% endcut %}

    the penalty is set to 0.1, 1 and 3 for the first, second and third features respectively. The penalty for all other features is set to 0.

- Set the penalty individually for each explicitly specified feature as a string (the number of features is n).

    {% cut "Format" %}

    ```
    "<feature index or name>:<penalty>,..,<feature index or name>:<penalty>"
    ```

    {% note info %}

    Spaces between values are not allowed.

    {% endnote %}

    {% endcut %}

    {% cut "These examples" %}

    `--first-feature-use-penalties` parameter:

    ```
    --first-feature-use-penalties "2:0.1,4:1.3"
    ```

    ```
    --first-feature-use-penalties "Feature2:0.1,Feature4:1.3"
    ```

    `--per-object-feature-penalties` parameter:

    ```
    --per-object-feature-penalties "2:0.1,4:1.3"
    ```

    ```
    --per-object-feature-penalties "Feature2:0.1,Feature4:1.3"
    ```

    {% endcut %}

    are identical, given that the name of the feature indexed 2 is <q>Feature2</q> and the name of the feature indexed 4 is <q>Feature4</q>.

- Set the penalty individually for each required feature as an array or a dictionary (the number of features is n).

    {% cut "Format" %}

    ```
    [<feature-penalty_0>, <feature-penalty_2>, .., <feature-penalty_n-1>]
    ```

    ```
    {"<feature index or name>":<penalty>, .., "<feature index or name>":<penalty>}
    ```

    {% endcut %}

    This format can be used if parameters are passed in a JSON file (see the `--params-file` parameter).

    {% cut "Examples" %}

    `--first-feature-use-penalties` parameter:

    ```json
    {
    "first_feature_use_penalties": {"Feature2":0.1,"Feature4":1.3}
    }
    ```

    ```json
    {
    "first_feature_use_penalties": {"2":0.1, "4":1.3}
    }
    ```

    ```json
    {
    "first_feature_use_penalties": [0.1,0.2,1.3,0.4,2.3]
    }
    ```

    `--per-object-feature-penalties` parameter:

    ```json
    {
    "per_object_feature_penalties": {"Feature2":0.1,"Feature4":1.3}
    }
    ```

    ```json
    {
    "per_object_feature_penalties": {"2":0.1, "4":1.3}
    }
    ```

    ```json
    {
    "per_object_feature_penalties": [0.1,0.2,1.3,0.4,2.3]
    }
    ```

    {% endcut %}
