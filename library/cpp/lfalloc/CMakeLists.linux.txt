
# This file was gererated by the build system used internally in the Yandex monorepo.
# Only simple modifications are allowed (adding source-files to targets, adding simple properties
# like target_include_directories). These modifications will be ported to original
# ya.make files by maintainers. Any complex modifications which can't be ported back to the
# original buildsystem will not be accepted.



add_library(library-cpp-lfalloc)
target_link_libraries(library-cpp-lfalloc PUBLIC
  contrib-libs-cxxsupp
  cpp-malloc-api
)
target_sources(library-cpp-lfalloc PRIVATE
  ${CMAKE_SOURCE_DIR}/library/cpp/lfalloc/lf_allocX64.cpp
)
