
# This file was gererated by the build system used internally in the Yandex monorepo.
# Only simple modifications are allowed (adding source-files to targets, adding simple properties
# like target_include_directories). These modifications will be ported to original
# ya.make files by maintainers. Any complex modifications which can't be ported back to the
# original buildsystem will not be accepted.



add_library(private-libs-index_range)
target_link_libraries(private-libs-index_range PUBLIC
  contrib-libs-cxxsupp
  yutil
)
target_sources(private-libs-index_range PRIVATE
  ${CMAKE_SOURCE_DIR}/catboost/private/libs/index_range/index_range.cpp
)
