
# This file was gererated by the build system used internally in the Yandex monorepo.
# Only simple modifications are allowed (adding source-files to targets, adding simple properties
# like target_include_directories). These modifications will be ported to original
# ya.make files by maintainers. Any complex modifications which can't be ported back to the
# original buildsystem will not be accepted.



add_library(private-libs-distributed)
target_link_libraries(private-libs-distributed PUBLIC
  contrib-libs-cxxsupp
  yutil
  catboost-libs-data
  catboost-libs-helpers
  catboost-libs-metrics
  private-libs-algo
  libs-algo-approx_calcer
  private-libs-algo_helpers
  private-libs-index_range
  private-libs-options
  library-cpp-binsaver
  library-cpp-json
  library-cpp-par
)
target_sources(private-libs-distributed PRIVATE
  ${CMAKE_SOURCE_DIR}/catboost/private/libs/distributed/mappers.cpp
  ${CMAKE_SOURCE_DIR}/catboost/private/libs/distributed/master.cpp
  ${CMAKE_SOURCE_DIR}/catboost/private/libs/distributed/worker.cpp
)
