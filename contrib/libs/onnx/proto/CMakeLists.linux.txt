
# This file was gererated by the build system used internally in the Yandex monorepo.
# Only simple modifications are allowed (adding source-files to targets, adding simple properties
# like target_include_directories). These modifications will be ported to original
# ya.make files by maintainers. Any complex modifications which can't be ported back to the
# original buildsystem will not be accepted.



add_library(libs-onnx-proto)
target_include_directories(libs-onnx-proto PUBLIC
  ${CMAKE_BINARY_DIR}/contrib/libs/onnx
)
target_link_libraries(libs-onnx-proto PUBLIC
  contrib-libs-cxxsupp
  yutil
  contrib-libs-protobuf
)
target_proto_messages(libs-onnx-proto PRIVATE
  ${CMAKE_SOURCE_DIR}/contrib/libs/onnx/onnx/onnx_data.proto
  ${CMAKE_SOURCE_DIR}/contrib/libs/onnx/onnx/onnx_ml.proto
  ${CMAKE_SOURCE_DIR}/contrib/libs/onnx/onnx/onnx_operators_ml.proto
)
target_proto_addincls(libs-onnx-proto
  ./contrib/libs/onnx
  ${CMAKE_SOURCE_DIR}/contrib/libs/onnx
  ${CMAKE_BINARY_DIR}
  ${CMAKE_SOURCE_DIR}
  ${CMAKE_SOURCE_DIR}/contrib/libs/onnx
  ${CMAKE_SOURCE_DIR}/contrib/libs/protobuf/src
  ${CMAKE_BINARY_DIR}
  ${CMAKE_SOURCE_DIR}/contrib/libs/protobuf/src
)
target_proto_outs(libs-onnx-proto
  --cpp_out=${CMAKE_BINARY_DIR}/contrib/libs/onnx
  --cpp_styleguide_out=${CMAKE_BINARY_DIR}/contrib/libs/onnx
)
