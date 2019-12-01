from .core import get_catboost_bin_module

_catboost = get_catboost_bin_module()
Tokenizer = _catboost.Tokenizer
Dictionary = _catboost.Dictionary
