LIBRARY()



CFLAGS(-DONNX_ML=1 -DONNX_NAMESPACE=onnx)

SRCS(
    coreml_helpers.cpp
    cpp_exporter.cpp
    export_helpers.cpp
    json_model_helpers.cpp
    model_exporter.cpp
    GLOBAL model_import.cpp
    onnx_helpers.cpp
    pmml_helpers.cpp
    python_exporter.cpp
)



PEERDIR(
    catboost/libs/model

    catboost/private/libs/ctr_description
    catboost/private/libs/labels
    contrib/libs/coreml
    contrib/libs/onnx
    library/cpp/resource
)

RESOURCE(
    catboost/libs/model/model_export/resources/apply_catboost_model.py catboost_model_export_python_model_applicator
    catboost/libs/model/model_export/resources/ctr_structs.py catboost_model_export_python_ctr_structs
    catboost/libs/model/model_export/resources/ctr_calcer.py catboost_model_export_python_ctr_calcer
    catboost/libs/model/model_export/resources/apply_catboost_model.cpp catboost_model_export_cpp_model_applicator
    catboost/libs/model/model_export/resources/ctr_structs.cpp catboost_model_export_cpp_ctr_structs
    catboost/libs/model/model_export/resources/ctr_calcer.cpp catboost_model_export_cpp_ctr_calcer
)

END()
