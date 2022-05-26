
# This file was gererated by the build system used internally in the Yandex monorepo.
# Only simple modifications are allowed (adding source-files to targets, adding simple properties
# like target_include_directories). These modifications will be ported to original
# ya.make files by maintainers. Any complex modifications which can't be ported back to the
# original buildsystem will not be accepted.



add_library(absl-strings-cord)
target_compile_options(absl-strings-cord PRIVATE
  -DNOMINMAX
)
target_include_directories(absl-strings-cord PUBLIC
  ${CMAKE_SOURCE_DIR}/contrib/restricted/abseil-cpp
)
target_include_directories(absl-strings-cord PRIVATE
  ${CMAKE_SOURCE_DIR}/contrib/restricted/abseil-cpp
)
target_link_libraries(absl-strings-cord PUBLIC
  contrib-libs-cxxsupp
  abseil-cpp-absl-algorithm
  abseil-cpp-absl-base
  base-internal-low_level_alloc
  base-internal-raw_logging
  base-internal-spinlock_wait
  base-internal-throw_delegate
  absl-base-log_severity
  abseil-cpp-absl-container
  abseil-cpp-absl-debugging
  absl-debugging-stacktrace
  absl-debugging-symbolize
  abseil-cpp-absl-demangle
  abseil-cpp-absl-functional
  abseil-cpp-absl-numeric
  profiling-internal-exponential_biased
  abseil-cpp-absl-strings
  strings-internal-absl_cord_internal
  strings-internal-absl_strings_internal
  strings-internal-cordz_functions
  strings-internal-cordz_handle
  strings-internal-cordz_info
  abseil-cpp-absl-synchronization
  absl-synchronization-internal
  abseil-cpp-absl-time
  absl-time-civil_time
  absl-time-time_zone
  abseil-cpp-absl-types
  absl-types-bad_optional_access
  abseil-cpp-absl-utility
)
target_sources(absl-strings-cord PRIVATE
  ${CMAKE_SOURCE_DIR}/contrib/restricted/abseil-cpp/absl/strings/cord.cc
)
