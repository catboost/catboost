
# This file was gererated by the build system used internally in the Yandex monorepo.
# Only simple modifications are allowed (adding source-files to targets, adding simple properties
# like target_include_directories). These modifications will be ported to original
# ya.make files by maintainers. Any complex modifications which can't be ported back to the
# original buildsystem will not be accepted.



add_library(random-internal-randen_round_keys)
target_compile_options(random-internal-randen_round_keys PRIVATE
  -DNOMINMAX
)
target_include_directories(random-internal-randen_round_keys PUBLIC
  ${CMAKE_SOURCE_DIR}/contrib/restricted/abseil-cpp
)
target_include_directories(random-internal-randen_round_keys PRIVATE
  ${CMAKE_SOURCE_DIR}/contrib/restricted/abseil-cpp
)
target_link_libraries(random-internal-randen_round_keys PUBLIC
  contrib-libs-cxxsupp
)
target_sources(random-internal-randen_round_keys PRIVATE
  ${CMAKE_SOURCE_DIR}/contrib/restricted/abseil-cpp/absl/random/internal/randen_round_keys.cc
)
