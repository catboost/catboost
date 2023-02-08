DLL_JAVA(catboost4j-spark-impl)

NO_WERROR()



SRCS(
    calc_fstr.cpp
    ctrs.cpp
    dataset_rows_reader.cpp
    data_provider_builders.cpp
    features_layout.cpp
    groupid.cpp
    jni_helpers.cpp
    master.cpp
    meta_info.cpp
    model.cpp
    model_application.cpp
    options_helper.cpp
    pairs.cpp
    quantized_features_info.cpp
    quantization.cpp
    GLOBAL spark_quantized.cpp
    string.cpp
    target.cpp
    native_impl.swg
    vector_output.cpp
    worker.cpp
)

EXTRADIR(bindings/swiglib)

PEERDIR(
    library/cpp/dbg_output
    library/cpp/grid_creator
    library/cpp/json
    library/cpp/par
    library/cpp/threading/atomic
    library/cpp/threading/local_executor
    catboost/libs/cat_feature
    catboost/libs/column_description
    catboost/libs/data
    catboost/libs/helpers
    catboost/libs/logging
    catboost/libs/model
    catboost/libs/model/model_export
    catboost/libs/train_lib
    catboost/private/libs/algo
    catboost/private/libs/app_helpers
    catboost/private/libs/data_util
    catboost/private/libs/data_types
    catboost/private/libs/distributed
    catboost/private/libs/labels
    catboost/private/libs/options
    catboost/private/libs/quantized_pool
)

IF (OS_WINDOWS)
    ALLOCATOR(J)
ELSE()
    ALLOCATOR(MIM)
ENDIF()

STRIP()


IF (USE_SYSTEM_JDK)
    CFLAGS(-I${JAVA_HOME}/include)
    IF(OS_DARWIN)
        CFLAGS(-I${JAVA_HOME}/include/darwin)
    ELSEIF(OS_LINUX)
        CFLAGS(-I${JAVA_HOME}/include/linux)
    ELSEIF(OS_WINDOWS)
        CFLAGS(-I${JAVA_HOME}/include/win32)
    ENDIF()
ELSE()
    IF (NOT OPENSOURCE OR AUTOCHECK)
        PEERDIR(contrib/libs/jdk)
    ELSEIF(EXPORT_CMAKE)
        PEERDIR(build/platform/java/jni)
    ELSE()
        # warning instead of an error to enable configure w/o specifying JAVA_HOME
        MESSAGE(WARNING System JDK required)
    ENDIF()
ENDIF()


# needed to ensure that compatible _Unwind_* functions are used
IF (NOT OS_WINDOWS)
    PEERDIR(contrib/libs/libunwind)
    IF (OS_LINUX)
        SET_APPEND(LDFLAGS "-Wl,--exclude-libs,ALL")
    ENDIF()
ENDIF()

END()
