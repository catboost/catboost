
# This file was gererated by the build system used internally in the Yandex monorepo.
# Only simple modifications are allowed (adding source-files to targets, adding simple properties
# like target_include_directories). These modifications will be ported to original
# ya.make files by maintainers. Any complex modifications which can't be ported back to the
# original buildsystem will not be accepted.



add_library(private-libs-ctr_description)
target_link_libraries(private-libs-ctr_description PUBLIC
  contrib-libs-cxxsupp
  yutil
  catboost-libs-helpers
  tools-enum_parser-enum_serialization_runtime
)
target_sources(private-libs-ctr_description PRIVATE
  ${CMAKE_SOURCE_DIR}/catboost/private/libs/ctr_description/ctr_type.cpp
)
generate_enum_serilization(private-libs-ctr_description
  ${CMAKE_SOURCE_DIR}/catboost/private/libs/ctr_description/ctr_type.h
  INCLUDE_HEADERS
  catboost/private/libs/ctr_description/ctr_type.h
)
