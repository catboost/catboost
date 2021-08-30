# catboost.save_model

```no-highlight
catboost.save_model(model,
                    model_path)
```

## {{ dl--purpose }} {#purpose}

{% include [reusage-r-save_model__purpose](../_includes/work_src/reusage-r/save_model__purpose.md) %}


{% include [r-r--fstr-not-saved-note](../_includes/work_src/reusage/r--fstr-not-saved-note.md) %}


## {{ dl--args }} {#arguments}
### model

#### Description

 The model to be saved.

**Default value**

 {{ r--required }}

### model_path

#### Description


The path to the resulting binary file with the model description.

Used for solving other machine learning problems (for instance, applying a model).


**Default value**

 {{ r--required }}

