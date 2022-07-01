
# This file was gererated by the build system used internally in the Yandex monorepo.
# Only simple modifications are allowed (adding source-files to targets, adding simple properties
# like target_include_directories). These modifications will be ported to original
# ya.make files by maintainers. Any complex modifications which can't be ported back to the
# original buildsystem will not be accepted.



add_library(random-internal-absl_random_internal_distribution_test_util)
target_compile_options(random-internal-absl_random_internal_distribution_test_util PRIVATE
  -DNOMINMAX
  -Wno-everything
)
target_include_directories(random-internal-absl_random_internal_distribution_test_util PUBLIC
  ${CMAKE_SOURCE_DIR}/contrib/restricted/abseil-cpp
)
target_link_libraries(random-internal-absl_random_internal_distribution_test_util PUBLIC
  contrib-libs-cxxsupp
  abseil-cpp-absl-base
  base-internal-raw_logging
  base-internal-spinlock_wait
  base-internal-throw_delegate
  absl-base-log_severity
  abseil-cpp-absl-numeric
  abseil-cpp-absl-strings
  strings-internal-absl_strings_internal
  strings-internal-str_format
)
target_sources(random-internal-absl_random_internal_distribution_test_util PRIVATE
  ${CMAKE_SOURCE_DIR}/contrib/restricted/abseil-cpp/absl/random/internal/chi_square.cc
  ${CMAKE_SOURCE_DIR}/contrib/restricted/abseil-cpp/absl/random/internal/distribution_test_util.cc
)
