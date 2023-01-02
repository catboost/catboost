
# This file was gererated by the build system used internally in the Yandex monorepo.
# Only simple modifications are allowed (adding source-files to targets, adding simple properties
# like target_include_directories). These modifications will be ported to original
# ya.make files by maintainers. Any complex modifications which can't be ported back to the
# original buildsystem will not be accepted.



add_library(cpp-threading-thread_local)
target_link_libraries(cpp-threading-thread_local PUBLIC
  contrib-libs-cxxsupp
  yutil
  cpp-threading-hot_swap
  cpp-threading-skip_list
  tools-enum_parser-enum_serialization_runtime
)
target_sources(cpp-threading-thread_local PRIVATE
  ${CMAKE_SOURCE_DIR}/library/cpp/threading/thread_local/thread_local.cpp
)
generate_enum_serilization(cpp-threading-thread_local
  ${CMAKE_SOURCE_DIR}/library/cpp/threading/thread_local/thread_local.h
  INCLUDE_HEADERS
  library/cpp/threading/thread_local/thread_local.h
)
