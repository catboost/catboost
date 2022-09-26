
# This file was gererated by the build system used internally in the Yandex monorepo.
# Only simple modifications are allowed (adding source-files to targets, adding simple properties
# like target_include_directories). These modifications will be ported to original
# ya.make files by maintainers. Any complex modifications which can't be ported back to the
# original buildsystem will not be accepted.



add_library(catboost-libs-overfitting_detector)
target_link_libraries(catboost-libs-overfitting_detector PUBLIC
  contrib-libs-cxxsupp
  yutil
  catboost-libs-helpers
  catboost-libs-logging
  catboost-libs-metrics
  private-libs-options
  library-cpp-statistics
  tools-enum_parser-enum_serialization_runtime
)
target_sources(catboost-libs-overfitting_detector PRIVATE
  ${CMAKE_SOURCE_DIR}/catboost/libs/overfitting_detector/error_tracker.cpp
  ${CMAKE_SOURCE_DIR}/catboost/libs/overfitting_detector/overfitting_detector.cpp
)
generate_enum_serilization(catboost-libs-overfitting_detector
  ${CMAKE_SOURCE_DIR}/catboost/libs/overfitting_detector/overfitting_detector.h
  INCLUDE_HEADERS
  catboost/libs/overfitting_detector/overfitting_detector.h
)
