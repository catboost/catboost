
# This file was gererated by the build system used internally in the Yandex monorepo.
# Only simple modifications are allowed (adding source-files to targets, adding simple properties
# like target_include_directories). These modifications will be ported to original
# ya.make files by maintainers. Any complex modifications which can't be ported back to the
# original buildsystem will not be accepted.



add_library(abseil-cpp-absl-strings)
target_compile_options(abseil-cpp-absl-strings PRIVATE
  -DNOMINMAX
)
target_include_directories(abseil-cpp-absl-strings PUBLIC
  ${CMAKE_SOURCE_DIR}/contrib/restricted/abseil-cpp
)
target_include_directories(abseil-cpp-absl-strings PRIVATE
  ${CMAKE_SOURCE_DIR}/contrib/restricted/abseil-cpp
)
target_link_libraries(abseil-cpp-absl-strings PUBLIC
  contrib-libs-cxxsupp
  abseil-cpp-absl-base
  base-internal-raw_logging
  base-internal-spinlock_wait
  base-internal-throw_delegate
  absl-base-log_severity
  abseil-cpp-absl-numeric
  strings-internal-absl_strings_internal
)
target_sources(abseil-cpp-absl-strings PRIVATE
  ${CMAKE_SOURCE_DIR}/contrib/restricted/abseil-cpp/absl/strings/ascii.cc
  ${CMAKE_SOURCE_DIR}/contrib/restricted/abseil-cpp/absl/strings/charconv.cc
  ${CMAKE_SOURCE_DIR}/contrib/restricted/abseil-cpp/absl/strings/escaping.cc
  ${CMAKE_SOURCE_DIR}/contrib/restricted/abseil-cpp/absl/strings/internal/charconv_bigint.cc
  ${CMAKE_SOURCE_DIR}/contrib/restricted/abseil-cpp/absl/strings/internal/charconv_parse.cc
  ${CMAKE_SOURCE_DIR}/contrib/restricted/abseil-cpp/absl/strings/internal/memutil.cc
  ${CMAKE_SOURCE_DIR}/contrib/restricted/abseil-cpp/absl/strings/match.cc
  ${CMAKE_SOURCE_DIR}/contrib/restricted/abseil-cpp/absl/strings/numbers.cc
  ${CMAKE_SOURCE_DIR}/contrib/restricted/abseil-cpp/absl/strings/str_cat.cc
  ${CMAKE_SOURCE_DIR}/contrib/restricted/abseil-cpp/absl/strings/str_replace.cc
  ${CMAKE_SOURCE_DIR}/contrib/restricted/abseil-cpp/absl/strings/str_split.cc
  ${CMAKE_SOURCE_DIR}/contrib/restricted/abseil-cpp/absl/strings/string_view.cc
  ${CMAKE_SOURCE_DIR}/contrib/restricted/abseil-cpp/absl/strings/substitute.cc
)
