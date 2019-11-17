from .core import get_catboost_bin_module, CatBoost, CatBoostError

_catboost = get_catboost_bin_module()


def to_polynom(model):
    if not isinstance(model, CatBoost):
        raise CatBoostError("Model should be CatBoost")
    return _catboost.to_polynom(model._object)


def to_polynom_string(model):
    if not isinstance(model, CatBoost):
        raise CatBoostError("Model should be CatBoost")
    return _catboost.to_polynom_string(model._object)
