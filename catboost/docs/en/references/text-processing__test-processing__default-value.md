# Text processing default JSON

```json
{
    "tokenizers" : [{
        "tokenizer_id" : "Space",
        "separator_type" : "ByDelimiter",
        "delimiter" : " "
    }],

    "dictionaries" : [{
        "dictionary_id" : "BiGram",
        "max_dictionary_size" : "50000",
        "occurrence_lower_bound" : "3",
        "gram_order" : "2"
    }, {
        "dictionary_id" : "Word",
        "max_dictionary_size" : "50000",
        "occurrence_lower_bound" : "3",
        "gram_order" : "1"
    }],

    "feature_processing" : {
        "default" : [{
            "dictionaries_names" : ["BiGram", "Word"],
            "feature_calcers" : ["BoW"],
            "tokenizers_names" : ["Space"]
        }, {
            "dictionaries_names" : ["Word"],
            "feature_calcers" : ["NaiveBayes"],
            "tokenizers_names" : ["Space"]
        }],
    }
}
```

