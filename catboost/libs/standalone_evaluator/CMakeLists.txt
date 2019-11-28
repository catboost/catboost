cmake_minimum_required(VERSION 2.8)

project(catboost_standalone_evaluator)
# ==============================================================================
# This example works only if flatbuffers headers and binaries are available
# ==============================================================================
add_definitions(-std=c++11)

function(compile_flatbuffers_schema_to_cpp SRC_FBS_DIR)
    get_filename_component(SRC_FBS ${SRC_FBS_DIR} NAME)
    string(REGEX REPLACE "\\.fbs$" "_generated.h" GEN_HEADER ${SRC_FBS})
    add_custom_command(
            OUTPUT ${GEN_HEADER}
            COMMAND flatc -c --gen-mutable
            -o "${CMAKE_CURRENT_SOURCE_DIR}"
            --reflect-names
            -I "${CMAKE_CURRENT_SOURCE_DIR}/../../.."
            -I "${CMAKE_CURRENT_SOURCE_DIR}/../model/flatbuffers"
            "${CMAKE_CURRENT_SOURCE_DIR}/${SRC_FBS_DIR}")
endfunction()

compile_flatbuffers_schema_to_cpp(../model/flatbuffers/features.fbs)
compile_flatbuffers_schema_to_cpp(../model/flatbuffers/ctr_data.fbs)
compile_flatbuffers_schema_to_cpp(../model/flatbuffers/model.fbs)
compile_flatbuffers_schema_to_cpp(../helpers/flatbuffers/guid.fbs)

set(SRCS
    example.cpp
    evaluator.cpp
    features_generated.h
    ctr_data_generated.h
    model_generated.h
    guid_generated.h)


add_executable(catboost_demo ${SRCS})
