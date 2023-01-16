# load

{% include [descriptions-load](../_includes/work_src/reusage-tokenizer/load.md) %}

## {{ dl--invoke-format }} {#call-format}

```
load(frequency_dict_path, 
     bpe_path=None)
```

## {{ dl--parameters }} {#parameters}

### frequency_dict_path

#### Description

The path to the input frequency based dictionary file.

**Data types**

{{ python-type--string }}

**Default value**

{{ python-type--string }}

### bpe_path

#### Description

The path to the input BPE dictionary file.

**Data types**

{{ python-type--string }} 

**Default value**

None (the dictionary is not loaded)

## {{ dl--output-format }} {#return-value}

_catboost.Dictionary

