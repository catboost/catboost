
# This file was gererated by the build system used internally in the Yandex monorepo.
# Only simple modifications are allowed (adding source-files to targets, adding simple properties
# like target_include_directories). These modifications will be ported to original
# ya.make files by maintainers. Any complex modifications which can't be ported back to the
# original buildsystem will not be accepted.



add_library(absl-debugging-leak_check)
target_compile_options(absl-debugging-leak_check PRIVATE
  -DNOMINMAX
  -Wno-everything
)
target_include_directories(absl-debugging-leak_check PUBLIC
  ${CMAKE_SOURCE_DIR}/contrib/restricted/abseil-cpp
)
target_link_libraries(absl-debugging-leak_check PUBLIC
  contrib-libs-cxxsupp
)
target_sources(absl-debugging-leak_check PRIVATE
  ${CMAKE_SOURCE_DIR}/contrib/restricted/abseil-cpp/absl/debugging/leak_check.cc
)
