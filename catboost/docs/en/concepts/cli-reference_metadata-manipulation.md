# Metadata manipulation

The {{ product }} models contain metadata (for example, the list of training parameters or user-defined data) in key-value format. Several operations are provided to manipulate the model's metadata.


## Get values of the selected keys {#get-metada}

### {{ dl__cli__execution-format }} {#execution-format}

```
catboost metadata get `-m` <path to the model> `--key` '<>' .. `--key` '<>' [`--dump-format` <output format>]
```

### {{ common-text__title__reference__parameters }} {#options}

### Output format {#output-format}

{% include [metadata-the-format-of-the-output-to-stdout-data](../_includes/work_src/reusage-common-phrases/the-format-of-the-output-to-stdout-data.md) %}

{% list tabs %}

- {{ cli__metadata__dump_format__plain }}

    Each line of the output contains tab-separated data on the key and its' value:

    ```
    key_1</t>value_1
    key_2</t>value_2
    ...
    key_N</t>value_N
    ```

    For example:
    ```
    model_name</t>catboost model
    params</t>{"detailed_profile":true,"used_ram_limit":18446744073709551615}
    ```

- {{ cli__metadata__dump_format__json }}

    A JSON array of metadata key-value pairs:

    ```
    {"model_name":"value_1","key_2":"value_2",..,"key_N":"value_N"}
    ```

    For example:
    ```
    {"model_name":"catboost model","GUITARMAN3":"3","params":"{\"detailed_profile\":true,\"has_time\":false}"}
    ```

{% endlist %}

## Get values of all keys {#dump-metadata}

### {{ dl__cli__execution-format }} {#execution-format}

```
catboost metadata dump `-m` <path to the model> [`--dump-format` <output format>]
```

### {{ common-text__title__reference__parameters }} {#options}

### -m, --model-file, --model-path

#### Description

The name of the resulting  files with the model description.

Used for solving other machine learning problems (for instance, applying a model) or defining the names of models in different output formats.

Corresponding file extensions are added to the given value if several output formats are defined in the `--model-format` parameter.

**{{ cli__params-table__title__default }}**

 {{ cli__required }}

### --dump-format

#### Description

The output format.

Possible values:
- Plain
- JSON

**{{ cli__params-table__title__default }}**

Plain

### Output {#output}

{% include [metadata-the-format-of-the-output-to-stdout-data](../_includes/work_src/reusage-common-phrases/the-format-of-the-output-to-stdout-data.md) %}

{% list tabs %}

- {{ cli__metadata__dump_format__plain }}

    Each line of the output contains tab-separated data on the key and its' value:

    ```
    key_1</t>value_1
    key_2</t>value_2
    ...
    key_N</t>value_N
    ```

    For example:
    ```
    model_name</t>catboost model
    params</t>{"detailed_profile":true,"used_ram_limit":18446744073709551615}
    ```

- {{ cli__metadata__dump_format__json }}

    A JSON array of metadata key-value pairs:

    ```
    {"model_name":"value_1","key_2":"value_2",..,"key_N":"value_N"}
    ```

    For example:
    ```
    {"model_name":"catboost model","GUITARMAN3":"3","params":"{\"detailed_profile\":true,\"has_time\":false}"}
    ```

{% endlist %}

## Set a key-value pair {#set-metadata}

### {{ dl__cli__execution-format }} {#execution-format}

```
catboost metadata set `-m` <path to the model> `--key` '<>' `--value` 'VALUE' []
```

### {{ common-text__title__reference__parameters }} {#options}

### -m, --model-file, --model-path

#### Description

The name of the resulting files with the model description.

Used for solving other machine learning problems (for instance, applying a model) or defining the names of models in different output formats.

Corresponding file extensions are added to the given value if several output formats are defined in the `--model-format` parameter.

**{{ cli__params-table__title__default }}**

 {{ cli__required }}

### --key

#### Description

The name of the key.

**{{ cli__params-table__title__default }}**

 Required key


### --value

#### Description

The value that the key should be set to.

The given value overwrites existing data if the specified key matches an existing one.


**{{ cli__params-table__title__default }}**

 Required key


### -o

#### Description

The path to the output model.

If defined, the key-value pair and the initial model's key-value pairs are written to the specified model and the input model is not affected by the operation.

**{{ cli__params-table__title__default }}**

 Input model (the values of existing keys are overridden)

