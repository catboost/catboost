
# This file was gererated by the build system used internally in the Yandex monorepo.
# Only simple modifications are allowed (adding source-files to targets, adding simple properties
# like target_include_directories). These modifications will be ported to original
# ya.make files by maintainers. Any complex modifications which can't be ported back to the
# original buildsystem will not be accepted.



add_library(contrib-libs-coreml)
target_link_libraries(contrib-libs-coreml PUBLIC
  contrib-libs-cxxsupp
  yutil
  contrib-libs-protobuf
)
target_proto_messages(contrib-libs-coreml PRIVATE
  ${CMAKE_SOURCE_DIR}/contrib/libs/coreml/ArrayFeatureExtractor.proto
  ${CMAKE_SOURCE_DIR}/contrib/libs/coreml/CategoricalMapping.proto
  ${CMAKE_SOURCE_DIR}/contrib/libs/coreml/DataStructures.proto
  ${CMAKE_SOURCE_DIR}/contrib/libs/coreml/DictVectorizer.proto
  ${CMAKE_SOURCE_DIR}/contrib/libs/coreml/FeatureTypes.proto
  ${CMAKE_SOURCE_DIR}/contrib/libs/coreml/FeatureVectorizer.proto
  ${CMAKE_SOURCE_DIR}/contrib/libs/coreml/GLMClassifier.proto
  ${CMAKE_SOURCE_DIR}/contrib/libs/coreml/GLMRegressor.proto
  ${CMAKE_SOURCE_DIR}/contrib/libs/coreml/Identity.proto
  ${CMAKE_SOURCE_DIR}/contrib/libs/coreml/Imputer.proto
  ${CMAKE_SOURCE_DIR}/contrib/libs/coreml/Model.proto
  ${CMAKE_SOURCE_DIR}/contrib/libs/coreml/NeuralNetwork.proto
  ${CMAKE_SOURCE_DIR}/contrib/libs/coreml/Normalizer.proto
  ${CMAKE_SOURCE_DIR}/contrib/libs/coreml/OneHotEncoder.proto
  ${CMAKE_SOURCE_DIR}/contrib/libs/coreml/SVM.proto
  ${CMAKE_SOURCE_DIR}/contrib/libs/coreml/Scaler.proto
  ${CMAKE_SOURCE_DIR}/contrib/libs/coreml/TreeEnsemble.proto
)
target_proto_addincls(contrib-libs-coreml
  ./
  ${CMAKE_SOURCE_DIR}/
  ${CMAKE_BINARY_DIR}
  ${CMAKE_SOURCE_DIR}
  ${CMAKE_SOURCE_DIR}/contrib/libs/protobuf/src
  ${CMAKE_BINARY_DIR}
  ${CMAKE_SOURCE_DIR}/contrib/libs/protobuf/src
)
target_proto_outs(contrib-libs-coreml
  --cpp_out=${CMAKE_BINARY_DIR}/
  --cpp_styleguide_out=${CMAKE_BINARY_DIR}/
)
