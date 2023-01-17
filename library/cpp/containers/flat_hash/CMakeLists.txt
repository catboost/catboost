
# This file was gererated by the build system used internally in the Yandex monorepo.
# Only simple modifications are allowed (adding source-files to targets, adding simple properties
# like target_include_directories). These modifications will be ported to original
# ya.make files by maintainers. Any complex modifications which can't be ported back to the
# original buildsystem will not be accepted.



add_library(cpp-containers-flat_hash)
target_link_libraries(cpp-containers-flat_hash PUBLIC
  contrib-libs-cxxsupp
  yutil
)
target_sources(cpp-containers-flat_hash PRIVATE
  ${CMAKE_SOURCE_DIR}/library/cpp/containers/flat_hash/flat_hash.cpp
  ${CMAKE_SOURCE_DIR}/library/cpp/containers/flat_hash/lib/concepts/container.cpp
  ${CMAKE_SOURCE_DIR}/library/cpp/containers/flat_hash/lib/concepts/iterator.cpp
  ${CMAKE_SOURCE_DIR}/library/cpp/containers/flat_hash/lib/concepts/size_fitter.cpp
  ${CMAKE_SOURCE_DIR}/library/cpp/containers/flat_hash/lib/concepts/value_marker.cpp
  ${CMAKE_SOURCE_DIR}/library/cpp/containers/flat_hash/lib/containers.cpp
  ${CMAKE_SOURCE_DIR}/library/cpp/containers/flat_hash/lib/expanders.cpp
  ${CMAKE_SOURCE_DIR}/library/cpp/containers/flat_hash/lib/iterator.cpp
  ${CMAKE_SOURCE_DIR}/library/cpp/containers/flat_hash/lib/map.cpp
  ${CMAKE_SOURCE_DIR}/library/cpp/containers/flat_hash/lib/probings.cpp
  ${CMAKE_SOURCE_DIR}/library/cpp/containers/flat_hash/lib/set.cpp
  ${CMAKE_SOURCE_DIR}/library/cpp/containers/flat_hash/lib/size_fitters.cpp
  ${CMAKE_SOURCE_DIR}/library/cpp/containers/flat_hash/lib/table.cpp
  ${CMAKE_SOURCE_DIR}/library/cpp/containers/flat_hash/lib/value_markers.cpp
)
