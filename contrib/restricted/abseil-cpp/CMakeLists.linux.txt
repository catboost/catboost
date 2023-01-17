
# This file was gererated by the build system used internally in the Yandex monorepo.
# Only simple modifications are allowed (adding source-files to targets, adding simple properties
# like target_include_directories). These modifications will be ported to original
# ya.make files by maintainers. Any complex modifications which can't be ported back to the
# original buildsystem will not be accepted.



add_library(contrib-restricted-abseil-cpp INTERFACE)
target_link_libraries(contrib-restricted-abseil-cpp INTERFACE
  abseil-cpp-absl-algorithm
  abseil-cpp-absl-base
  base-internal-low_level_alloc
  base-internal-raw_logging
  base-internal-scoped_set_env
  base-internal-spinlock_wait
  base-internal-strerror
  base-internal-throw_delegate
  absl-base-log_severity
  abseil-cpp-absl-city
  abseil-cpp-absl-container
  container-internal-absl_hashtablez_sampler
  container-internal-raw_hash_set
  abseil-cpp-absl-debugging
  absl-debugging-failure_signal_handler
  absl-debugging-internal
  absl-debugging-leak_check
  absl-debugging-leak_check_disable
  absl-debugging-stacktrace
  absl-debugging-symbolize
  abseil-cpp-absl-demangle
  abseil-cpp-absl-flags
  absl-flags-commandlineflag
  flags-internal-commandlineflag
  flags-internal-flag
  flags-internal-private_handle_accessor
  flags-internal-program_name
  flags-internal-usage
  absl-flags-marshalling
  absl-flags-parse
  absl-flags-reflection
  absl-flags-usage
  absl-flags-usage_config
  abseil-cpp-absl-functional
  abseil-cpp-absl-hash
  absl-hash-internal
  abseil-cpp-absl-memory
  abseil-cpp-absl-meta
  abseil-cpp-absl-numeric
  profiling-internal-exponential_biased
  profiling-internal-periodic_sampler
  absl-random-absl_random_distributions
  random-internal-absl_random_internal_distribution_test_util
  random-internal-pool_urbg
  random-internal-randen
  random-internal-randen_detect
  random-internal-randen_hwaes
  random-internal-randen_round_keys
  random-internal-randen_slow
  random-internal-seed_material
  absl-random-seed_gen_exception
  absl-random-seed_sequences
  abseil-cpp-absl-status
  absl-status-statusor
  abseil-cpp-absl-strings
  absl-strings-cord
  strings-internal-absl_cord_internal
  strings-internal-absl_strings_internal
  strings-internal-cordz_functions
  strings-internal-cordz_handle
  strings-internal-cordz_info
  strings-internal-cordz_sample_token
  strings-internal-str_format
  abseil-cpp-absl-synchronization
  absl-synchronization-internal
  abseil-cpp-absl-time
  absl-time-civil_time
  absl-time-time_zone
  abseil-cpp-absl-types
  absl-types-bad_any_cast
  absl-types-bad_optional_access
  absl-types-bad_variant_access
  absl-types-internal
  abseil-cpp-absl-utility
)
