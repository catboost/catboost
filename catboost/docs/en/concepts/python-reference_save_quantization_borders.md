# save_quantization_borders

{% include [pool__save_quantization_borders-pool__save_quantization_borders_div](../_includes/work_src/reusage-python/pool__save_quantization_borders_div.md) %}

## {{ dl--invoke-format }} {#call-format}

```python
save_quantization_borders(output_file)
```

## {{ dl--parameters }} {#parameters}

### output_file

#### Description

The name of the output file to save borders used in the numeric features' quantization to.

{% include [pool__save_quantization_borders-pool__save_quantization_borders_ref_to_format](../_includes/work_src/reusage-python/pool__save_quantization_borders_ref_to_format.md) %}

**Possible types**

{{ python-type--string }}

**Default value**

Required parameter

## {{ input_data__title__example }} {#example}

The following example shows how to save borders used in numeric features' quantization in the training dataset to a file (`borders.dat`) and then use them for the evaluation dataset.

```python
from catboost import Pool, CatBoostRegressor

train_data = [[1, 4, 5, 6],
              [4, 5, 6, 7],
              [30, 40, 50, 60]]

train_labels = [10, 20, 30]

eval_data = [[2, 4, 6, 8],
             [1, 4, 50, 60]]

eval_labels = [20, 30]

train_dataset = Pool(train_data, train_labels)
eval_dataset = Pool(eval_data, eval_labels)

train_dataset.quantize()
train_dataset.save_quantization_borders("borders.dat")
eval_dataset.quantize(input_borders="borders.dat")
```

Contents of the output `borders.dat` file:
```no-highlight
0	2.5
0	17
1	4.5
1	22.5
2	5.5
2	28
3	6.5
3	33.5
```
