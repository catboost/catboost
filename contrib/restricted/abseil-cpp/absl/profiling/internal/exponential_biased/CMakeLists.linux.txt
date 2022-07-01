
# This file was gererated by the build system used internally in the Yandex monorepo.
# Only simple modifications are allowed (adding source-files to targets, adding simple properties
# like target_include_directories). These modifications will be ported to original
# ya.make files by maintainers. Any complex modifications which can't be ported back to the
# original buildsystem will not be accepted.



add_library(profiling-internal-exponential_biased)
target_compile_options(profiling-internal-exponential_biased PRIVATE
  -DNOMINMAX
  -Wno-everything
)
target_include_directories(profiling-internal-exponential_biased PUBLIC
  ${CMAKE_SOURCE_DIR}/contrib/restricted/abseil-cpp
)
target_link_libraries(profiling-internal-exponential_biased PUBLIC
  contrib-libs-cxxsupp
)
target_sources(profiling-internal-exponential_biased PRIVATE
  ${CMAKE_SOURCE_DIR}/contrib/restricted/abseil-cpp/absl/profiling/internal/exponential_biased.cc
)
