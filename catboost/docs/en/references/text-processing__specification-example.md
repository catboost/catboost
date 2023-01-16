# Text processing JSON specification example

```json
"text_processing_options" : {
    "tokenizers" : [{
        "tokenizer_id" : "Space",
        "delimiter" : " ",
        "lowercasing" : "true"
    }],

    "dictionaries" : [{
        "dictionary_id" : "BiGram",
        "gram_order" : "2"
    }, {
        "dictionary_id" : "Word",
        "gram_order" : "1"
    }],

    "feature_processing" : {
        "default" : [{
            "dictionaries_names" : ["Word"],
            "feature_calcers" : ["BoW"],
            "tokenizers_names" : ["Space"]
        }],
        
        "1" : [{
            "tokenizers_names" : ["Space"],
            "dictionaries_names" : ["BiGram", "Word"],
            "feature_calcers" : ["BoW"]
        }, {
            "tokenizers_names" : ["Space"],
            "dictionaries_names" : ["Word"],
            "feature_calcers" : ["NaiveBayes"]
        }]
    }
}
```

In this example:
- A single split-by-delimiter tokenizer is specified. It lowercases tokens after splitting.
- Two dictionaries: unigram (identified <q>Word</q>) and bigram (identified <q>BiGram</q>).
- Two feature calcers are specified for the second text feature:
    - {{ dictionary__feature-calcers__BoW }}, which uses the <q>BiGram</q> and <q>Word</q> dictionaries.
    - {{ dictionary__feature-calcers__NaiveBayes }}, which uses the <q>Word</q> dictionary.
    
- A single feature calcer is specified for all other text features: {{ dictionary__feature-calcers__BoW }}, which uses the <q>Word</q> dictionary.

