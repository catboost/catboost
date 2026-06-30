# distutils: language = c++
# coding: utf-8
# cython: wraparound=False

from catboost.libs.carry_model.cython cimport *


cpdef _carry_by_name(model, factor_names, factor_values):
    cdef TFullModel tmp = CarryModelByName(
        dereference((<_CatBoost>model).__model),
        py_to_tvector[TString](factor_names),
        py_to_tvector_tvector[double](factor_values)
    )
    (<_CatBoost>model).model_blob = None
    (<_CatBoost>model).__model.Swap(tmp)

cpdef _carry_by_index(model, factor_indexes, factor_values):
    cdef TFullModel tmp = CarryModelByFlatIndex(
        dereference((<_CatBoost>model).__model),
        py_to_tvector[int](factor_indexes),
        py_to_tvector_tvector[double](factor_values)
    )
    (<_CatBoost>model).model_blob = None
    (<_CatBoost>model).__model.Swap(tmp)

cpdef _uplift_by_name(model, factor_names, factor_base_values, factor_next_values):
    cdef TFullModel tmp = UpliftModelByName(
        dereference((<_CatBoost>model).__model),
        py_to_tvector[TString](factor_names),
        py_to_tvector[double](factor_base_values),
        py_to_tvector[double](factor_next_values)
    )
    (<_CatBoost>model).model_blob = None
    (<_CatBoost>model).__model.Swap(tmp)

cpdef _uplift_by_index(model, factor_indexes, factor_base_values, factor_next_values):
    cdef TFullModel tmp = UpliftModelByFlatIndex(
        dereference((<_CatBoost>model).__model),
        py_to_tvector[int](factor_indexes),
        py_to_tvector[double](factor_base_values),
        py_to_tvector[double](factor_next_values)
    )
    (<_CatBoost>model).model_blob = None
    (<_CatBoost>model).__model.Swap(tmp)
