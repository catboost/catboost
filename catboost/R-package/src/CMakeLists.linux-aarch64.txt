
# This file was gererated by the build system used internally in the Yandex monorepo.
# Only simple modifications are allowed (adding source-files to targets, adding simple properties
# like target_include_directories). These modifications will be ported to original
# ya.make files by maintainers. Any complex modifications which can't be ported back to the
# original buildsystem will not be accepted.



add_library(catboostr SHARED)
target_include_directories(catboostr PRIVATE
  ${CMAKE_SOURCE_DIR}/contrib/libs/r-lang
)
target_link_libraries(catboostr PUBLIC
  contrib-libs-cxxsupp
  yutil
  cpp-malloc-mimalloc
  catboost-libs-cat_feature
  catboost-libs-data
  catboost-libs-eval_result
  catboost-libs-fstr
  libs-gpu_config-maybe_have_cuda
  catboost-libs-logging
  catboost-libs-model
  catboost-libs-train_lib
  private-libs-algo
  private-libs-data_util
  private-libs-documents_importance
  private-libs-init
  private-libs-options
)
target_sources(catboostr PRIVATE
  ${CMAKE_SOURCE_DIR}/catboost/R-package/src/catboostr.cpp
  ${CMAKE_SOURCE_DIR}/catboost/R-package/src/init.c
)
use_export_script(catboostr
  ${CMAKE_SOURCE_DIR}/catboost/R-package/src/catboostr.exports
)
vcs_info(catboostr)
