
# This file was gererated by the build system used internally in the Yandex monorepo.
# Only simple modifications are allowed (adding source-files to targets, adding simple properties
# like target_include_directories). These modifications will be ported to original
# ya.make files by maintainers. Any complex modifications which can't be ported back to the
# original buildsystem will not be accepted.



add_library(cpp-token-lite)
target_link_libraries(cpp-token-lite PUBLIC
  contrib-libs-cxxsupp
  yutil
  tools-enum_parser-enum_serialization_runtime
  library-cpp-langmask
)
target_sources(cpp-token-lite PRIVATE
  ${CMAKE_SOURCE_DIR}/library/cpp/token/accent.cpp
  ${CMAKE_SOURCE_DIR}/library/cpp/token/nlptypes.cpp
  ${CMAKE_SOURCE_DIR}/library/cpp/token/token_util.cpp
)
generate_enum_serilization(cpp-token-lite
  ${CMAKE_SOURCE_DIR}/library/cpp/token/nlptypes.h
  INCLUDE_HEADERS
  library/cpp/token/nlptypes.h
)
generate_enum_serilization(cpp-token-lite
  ${CMAKE_SOURCE_DIR}/library/cpp/token/formtype.h
  INCLUDE_HEADERS
  library/cpp/token/formtype.h
)
