
# This file was gererated by the build system used internally in the Yandex monorepo.
# Only simple modifications are allowed (adding source-files to targets, adding simple properties
# like target_include_directories). These modifications will be ported to original
# ya.make files by maintainers. Any complex modifications which can't be ported back to the
# original buildsystem will not be accepted.


find_package(JNITarget REQUIRED)

add_library(catboost4j-prediction SHARED)
target_link_libraries(catboost4j-prediction PUBLIC
  contrib-libs-cxxsupp
  yutil
  cpp-malloc-mimalloc
  catboost-libs-helpers
  catboost-libs-model
  JNITarget::jni
)
target_sources(catboost4j-prediction PRIVATE
  ${CMAKE_SOURCE_DIR}/catboost/jvm-packages/catboost4j-prediction/src/native_impl/ai_catboost_CatBoostJNIImpl.cpp
)
vcs_info(catboost4j-prediction)
