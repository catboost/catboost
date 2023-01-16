# save

{% include [descriptions-save](../_includes/work_src/reusage-tokenizer/save.md) %}


## {{ dl--invoke-format }} {#call-format}

```
save(frequency_dict_path, 
     bpe_path=None)
```

## {{ dl--parameters }} {#parameters}

### frequency_dict_path

#### Description

The path to the output file with the [frequency based dictionary](output-data_frequency-based-dict.md).

**Data types**

{{ loss-functions__params__q__default }}

**Default value**

{{ loss-functions__params__q__default }}

### bpe_path

#### Description

The path to the output file with the [BPE dictionary](output-data_bpe-dict.md).

**Data types**

{{ loss-functions__params__q__default }}

**Default value**

None (the dictionary is not saved)

## {{ dl--output-format }} {#return-value}

_catboost.Dictionary

