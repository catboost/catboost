# Baseline

#### {{ input_data__title__contains }}

Baseline values for objects in the dataset.

#### {{ input_data__title__specification }}

- List the baseline value for each object on a new line.
- Use a tab as the delimiter to separate the columns on a line.
- Use the {{ prediction-type--RawFormulaVal }} prediction type of the [model values](output-data_model-value-output.md) as the baseline. Other prediction types are not supported.

#### {{ input_data__title__row-format }}

- The first line contains the description of data in the corresponding column.

  Format:

  {% list tabs %}

    - Non multiclassification methods


      ```
          RawFormulaVal[<\t><other tab-separated columns>]
      ```

      `other tab-separated columns` can be set for auxiliary data (for example, for {{ cd-file__col-type__SampleId }}).

    - Multiclassification methods

      ```
          RawFormulaVal:Class=<class_name_1><\t>...<\t>RawFormulaVal:Class=<class_name_N>[<\t><other tab-separated columns>]
      ```

      `class_name_1`...`class_name_N` are the names of corresponding classes. The order must match the one specified in the `--classes-count` parameter and in the model.

  {% endlist %}

- All the other lines contain baseline values for the objects of the input dataset. Specify the baseline value for the `i`-th object on the line numbered `i+1`.

  Format:

  {% list tabs %}

    - Non multiclassification methods

      ```
          <value>[<\t><other tab-separated data>]
      ```

    - Multiclassification methods

      ```
          <value for class_name_1><\t>...<\t><value for class_name_N>[<\t><other tab-separated data]
      ```

      `class_name_1`...`class_name_N` are the names of corresponding classes.

  {% endlist %}


#### {{ input_data__title__example }}

```
RawFormulaVal
-2.036136239767074585e-01
-2.093112945556640625e+00
-1.437631964683532715e+00
9.149890393018722534e-02
```
