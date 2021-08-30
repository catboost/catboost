
- {% include [penalties-format-cli__first-feature-use-penalties__desc__weight-for-each-feature](../reusage-cli/cli__first-feature-use-penalties__desc__weight-for-each-feature.md) %}

    In this example

    `first_feature_use_penalties` parameter:

    ```
    first_feature_use_penalties = "(0.1,1,3)"
    ```

    `per_object_feature_penalties` parameter:

    ```
    per_object_feature_penalties = "(0.1,1,3)"
    ```

    {% note info %}

    Spaces between values are not allowed.

    {% endnote %}

    the multiplication weight is set to 0.1, 1 and 3 for the first, second and third features respectively. The multiplication weight for all other features is set to 1.

- {% include [penalties-format-cli__ffirst-feature-use-penalties__formats__individually-for-required-features](../reusage-cli/cli__ffirst-feature-use-penalties__formats__individually-for-required-features.md) %}

    These examples `first_feature_use_penalties` parameter:

    ```
    first_feature_use_penalties = "2:1.1,4:0.1"
    ```

    ```
    first_feature_use_penalties = "Feature2:1.1,Feature4:0.1"
    ```

    `per_object_feature_penalties` parameter:

    ```
    per_object_feature_penalties = "2:1.1,4:0.1"
    ```

    ```
    per_object_feature_penalties = "Feature2:1.1,Feature4:0.1"
    ```
    are identical, given that the name of the feature indexed 2 is <q>Feature2</q> and the name of the feature indexed 4 is <q>Feature4</q>.

- {% include [penalties-format-cli__first-feature-use-penalties__arra-or-dict](../reusage-cli/cli__first-feature-use-penalties__arra-or-dict.md) %}

    Array examples.

    `first_feature_use_penalties` parameter:

    ```
    first_feature_use_penalties = [0.1, 1, 3]
    ```

    `per_object_feature_penalties` parameter:

    ```
    per_object_feature_penalties = [0.1, 1, 3]
    ```

    These dictionary examples

    `first_feature_use_penalties` parameter:

    ```python
    first_feature_use_penalties = {"Feature2":1.1,"Feature4":0.1}
    ```

    ```python
    first_feature_use_penalties = {"2":1.1, "4":0.1}
    ```

    `per_object_feature_penalties` parameter:

    ```python
    per_object_feature_penalties = {"Feature2":1.1,"Feature4":0.1}
    ```

    ```python
    per_object_feature_penalties = {"2":1.1, "4":0.1}
    ```

    are identical, given that the name of the feature indexed 2 is <q>Feature2</q> and the name of the feature indexed 4 is <q>Feature4</q>.

