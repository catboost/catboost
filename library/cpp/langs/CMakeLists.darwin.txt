
# This file was gererated by the build system used internally in the Yandex monorepo.
# Only simple modifications are allowed (adding source-files to targets, adding simple properties
# like target_include_directories). These modifications will be ported to original
# ya.make files by maintainers. Any complex modifications which can't be ported back to the
# original buildsystem will not be accepted.



add_library(library-cpp-langs)
target_link_libraries(library-cpp-langs PUBLIC
  contrib-libs-cxxsupp
  yutil
  cpp-digest-lower_case
)
target_sources(library-cpp-langs PRIVATE
  ${CMAKE_SOURCE_DIR}/library/cpp/langs/generated/uniscripts.cpp
  ${CMAKE_SOURCE_DIR}/library/cpp/langs/langs.cpp
  ${CMAKE_SOURCE_DIR}/library/cpp/langs/scripts.cpp
)
