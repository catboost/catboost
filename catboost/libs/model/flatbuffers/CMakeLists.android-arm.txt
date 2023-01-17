
# This file was gererated by the build system used internally in the Yandex monorepo.
# Only simple modifications are allowed (adding source-files to targets, adding simple properties
# like target_include_directories). These modifications will be ported to original
# ya.make files by maintainers. Any complex modifications which can't be ported back to the
# original buildsystem will not be accepted.



add_library(libs-model-flatbuffers)
target_link_libraries(libs-model-flatbuffers PUBLIC
  contrib-libs-cxxsupp
  yutil
  contrib-libs-flatbuffers
)
target_sources(libs-model-flatbuffers PRIVATE
  ${CMAKE_BINARY_DIR}/catboost/libs/model/flatbuffers/features.fbs.cpp
  ${CMAKE_BINARY_DIR}/catboost/libs/model/flatbuffers/ctr_data.fbs.cpp
  ${CMAKE_BINARY_DIR}/catboost/libs/model/flatbuffers/model.fbs.cpp
)
target_fbs_source(libs-model-flatbuffers
  PRIVATE
  ${CMAKE_SOURCE_DIR}/catboost/libs/model/flatbuffers/features.fbs
  -I
  ${CMAKE_BINARY_DIR}
  -I
  ${CMAKE_SOURCE_DIR}
)
target_fbs_source(libs-model-flatbuffers
  PRIVATE
  ${CMAKE_SOURCE_DIR}/catboost/libs/model/flatbuffers/ctr_data.fbs
  -I
  ${CMAKE_BINARY_DIR}
  -I
  ${CMAKE_SOURCE_DIR}
)
target_fbs_source(libs-model-flatbuffers
  PRIVATE
  ${CMAKE_SOURCE_DIR}/catboost/libs/model/flatbuffers/model.fbs
  -I
  ${CMAKE_BINARY_DIR}
  -I
  ${CMAKE_SOURCE_DIR}
)
