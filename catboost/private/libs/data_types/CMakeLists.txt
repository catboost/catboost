
# This file was gererated by the build system used internally in the Yandex monorepo.
# Only simple modifications are allowed (adding source-files to targets, adding simple properties
# like target_include_directories). These modifications will be ported to original
# ya.make files by maintainers. Any complex modifications which can't be ported back to the
# original buildsystem will not be accepted.



add_library(private-libs-data_types)
target_link_libraries(private-libs-data_types PUBLIC
  contrib-libs-cxxsupp
  yutil
  private-libs-index_range
  library-cpp-binsaver
  cpp-containers-dense_hash
)
target_sources(private-libs-data_types PRIVATE
  ${CMAKE_SOURCE_DIR}/catboost/private/libs/data_types/pair.cpp
  ${CMAKE_SOURCE_DIR}/catboost/private/libs/data_types/query.cpp
  ${CMAKE_SOURCE_DIR}/catboost/private/libs/data_types/text.cpp
)
