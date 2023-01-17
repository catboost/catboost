
# This file was gererated by the build system used internally in the Yandex monorepo.
# Only simple modifications are allowed (adding source-files to targets, adding simple properties
# like target_include_directories). These modifications will be ported to original
# ya.make files by maintainers. Any complex modifications which can't be ported back to the
# original buildsystem will not be accepted.



add_library(libs-lz4-generated)
target_link_libraries(libs-lz4-generated PUBLIC
  contrib-libs-cxxsupp
)
target_sources(libs-lz4-generated PRIVATE
  ${CMAKE_SOURCE_DIR}/contrib/libs/lz4/generated/lz4_10.cpp
  ${CMAKE_SOURCE_DIR}/contrib/libs/lz4/generated/lz4_11.cpp
  ${CMAKE_SOURCE_DIR}/contrib/libs/lz4/generated/lz4_12.cpp
  ${CMAKE_SOURCE_DIR}/contrib/libs/lz4/generated/lz4_13.cpp
  ${CMAKE_SOURCE_DIR}/contrib/libs/lz4/generated/lz4_14.cpp
  ${CMAKE_SOURCE_DIR}/contrib/libs/lz4/generated/lz4_15.cpp
  ${CMAKE_SOURCE_DIR}/contrib/libs/lz4/generated/lz4_16.cpp
  ${CMAKE_SOURCE_DIR}/contrib/libs/lz4/generated/lz4_17.cpp
  ${CMAKE_SOURCE_DIR}/contrib/libs/lz4/generated/lz4_18.cpp
  ${CMAKE_SOURCE_DIR}/contrib/libs/lz4/generated/lz4_19.cpp
  ${CMAKE_SOURCE_DIR}/contrib/libs/lz4/generated/lz4_20.cpp
  ${CMAKE_SOURCE_DIR}/contrib/libs/lz4/generated/lz4methods.cpp
)
