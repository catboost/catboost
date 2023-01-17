
# This file was gererated by the build system used internally in the Yandex monorepo.
# Only simple modifications are allowed (adding source-files to targets, adding simple properties
# like target_include_directories). These modifications will be ported to original
# ya.make files by maintainers. Any complex modifications which can't be ported back to the
# original buildsystem will not be accepted.


add_subdirectory(ut)

add_library(private-libs-options)
target_link_libraries(private-libs-options PUBLIC
  contrib-libs-cxxsupp
  yutil
  library-cpp-json
  catboost-libs-column_description
  catboost-libs-helpers
  catboost-libs-logging
  private-libs-ctr_description
  private-libs-data_util
  cpp-getopt-small
  library-cpp-grid_creator
  cpp-text_processing-dictionary
  cpp-text_processing-tokenizer
  tools-enum_parser-enum_serialization_runtime
)
target_sources(private-libs-options PRIVATE
  ${CMAKE_SOURCE_DIR}/catboost/private/libs/options/analytical_mode_params.cpp
  ${CMAKE_SOURCE_DIR}/catboost/private/libs/options/binarization_options.cpp
  ${CMAKE_SOURCE_DIR}/catboost/private/libs/options/boosting_options.cpp
  ${CMAKE_SOURCE_DIR}/catboost/private/libs/options/bootstrap_options.cpp
  ${CMAKE_SOURCE_DIR}/catboost/private/libs/options/cat_feature_options.cpp
  ${CMAKE_SOURCE_DIR}/catboost/private/libs/options/catboost_options.cpp
  ${CMAKE_SOURCE_DIR}/catboost/private/libs/options/check_train_options.cpp
  ${CMAKE_SOURCE_DIR}/catboost/private/libs/options/class_label_options.cpp
  ${CMAKE_SOURCE_DIR}/catboost/private/libs/options/cross_validation_params.cpp
  ${CMAKE_SOURCE_DIR}/catboost/private/libs/options/data_processing_options.cpp
  ${CMAKE_SOURCE_DIR}/catboost/private/libs/options/dataset_reading_params.cpp
  ${CMAKE_SOURCE_DIR}/catboost/private/libs/options/defaults_helper.cpp
  ${CMAKE_SOURCE_DIR}/catboost/private/libs/options/embedding_processing_options.cpp
  ${CMAKE_SOURCE_DIR}/catboost/private/libs/options/enum_helpers.cpp
  ${CMAKE_SOURCE_DIR}/catboost/private/libs/options/feature_eval_options.cpp
  ${CMAKE_SOURCE_DIR}/catboost/private/libs/options/feature_penalties_options.cpp
  ${CMAKE_SOURCE_DIR}/catboost/private/libs/options/features_select_options.cpp
  ${CMAKE_SOURCE_DIR}/catboost/private/libs/options/json_helper.cpp
  ${CMAKE_SOURCE_DIR}/catboost/private/libs/options/load_options.cpp
  ${CMAKE_SOURCE_DIR}/catboost/private/libs/options/loss_description.cpp
  ${CMAKE_SOURCE_DIR}/catboost/private/libs/options/metric_options.cpp
  ${CMAKE_SOURCE_DIR}/catboost/private/libs/options/model_based_eval_options.cpp
  ${CMAKE_SOURCE_DIR}/catboost/private/libs/options/monotone_constraints.cpp
  ${CMAKE_SOURCE_DIR}/catboost/private/libs/options/oblivious_tree_options.cpp
  ${CMAKE_SOURCE_DIR}/catboost/private/libs/options/option.cpp
  ${CMAKE_SOURCE_DIR}/catboost/private/libs/options/output_file_options.cpp
  ${CMAKE_SOURCE_DIR}/catboost/private/libs/options/overfitting_detector_options.cpp
  ${CMAKE_SOURCE_DIR}/catboost/private/libs/options/path_helpers.cpp
  ${CMAKE_SOURCE_DIR}/catboost/private/libs/options/parse_per_feature_options.cpp
  ${CMAKE_SOURCE_DIR}/catboost/private/libs/options/plain_options_helper.cpp
  ${CMAKE_SOURCE_DIR}/catboost/private/libs/options/pool_metainfo_options.cpp
  ${CMAKE_SOURCE_DIR}/catboost/private/libs/options/runtime_text_options.cpp
  ${CMAKE_SOURCE_DIR}/catboost/private/libs/options/runtime_embedding_options.cpp
  ${CMAKE_SOURCE_DIR}/catboost/private/libs/options/split_params.cpp
  ${CMAKE_SOURCE_DIR}/catboost/private/libs/options/system_options.cpp
  ${CMAKE_SOURCE_DIR}/catboost/private/libs/options/text_processing_options.cpp
)
generate_enum_serilization(private-libs-options
  ${CMAKE_SOURCE_DIR}/catboost/private/libs/options/enums.h
  INCLUDE_HEADERS
  catboost/private/libs/options/enums.h
)
generate_enum_serilization(private-libs-options
  ${CMAKE_SOURCE_DIR}/catboost/private/libs/options/json_helper.h
  INCLUDE_HEADERS
  catboost/private/libs/options/json_helper.h
)
