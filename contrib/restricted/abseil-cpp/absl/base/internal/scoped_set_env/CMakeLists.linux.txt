
# This file was gererated by the build system used internally in the Yandex monorepo.
# Only simple modifications are allowed (adding source-files to targets, adding simple properties
# like target_include_directories). These modifications will be ported to original
# ya.make files by maintainers. Any complex modifications which can't be ported back to the
# original buildsystem will not be accepted.



add_library(base-internal-scoped_set_env)
target_compile_options(base-internal-scoped_set_env PRIVATE
  -DNOMINMAX
  -Wno-everything
)
target_include_directories(base-internal-scoped_set_env PUBLIC
  ${CMAKE_SOURCE_DIR}/contrib/restricted/abseil-cpp
)
target_link_libraries(base-internal-scoped_set_env PUBLIC
  contrib-libs-cxxsupp
  base-internal-raw_logging
  absl-base-log_severity
)
target_sources(base-internal-scoped_set_env PRIVATE
  ${CMAKE_SOURCE_DIR}/contrib/restricted/abseil-cpp/absl/base/internal/scoped_set_env.cc
)
