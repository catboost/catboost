
# This file was gererated by the build system used internally in the Yandex monorepo.
# Only simple modifications are allowed (adding source-files to targets, adding simple properties
# like target_include_directories). These modifications will be ported to original
# ya.make files by maintainers. Any complex modifications which can't be ported back to the
# original buildsystem will not be accepted.



add_library(absl-flags-usage_config)
target_compile_options(absl-flags-usage_config PRIVATE
  -DNOMINMAX
  -Wno-everything
)
target_include_directories(absl-flags-usage_config PUBLIC
  ${CMAKE_SOURCE_DIR}/contrib/restricted/abseil-cpp
)
target_link_libraries(absl-flags-usage_config PUBLIC
  contrib-libs-cxxsupp
  abseil-cpp-absl-base
  base-internal-low_level_alloc
  base-internal-raw_logging
  base-internal-spinlock_wait
  base-internal-throw_delegate
  absl-base-log_severity
  abseil-cpp-absl-debugging
  absl-debugging-stacktrace
  absl-debugging-symbolize
  abseil-cpp-absl-demangle
  flags-internal-program_name
  abseil-cpp-absl-numeric
  abseil-cpp-absl-strings
  strings-internal-absl_strings_internal
  abseil-cpp-absl-synchronization
  absl-synchronization-internal
  abseil-cpp-absl-time
  absl-time-civil_time
  absl-time-time_zone
)
target_sources(absl-flags-usage_config PRIVATE
  ${CMAKE_SOURCE_DIR}/contrib/restricted/abseil-cpp/absl/flags/usage_config.cc
)
