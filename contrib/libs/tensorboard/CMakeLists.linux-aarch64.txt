
# This file was gererated by the build system used internally in the Yandex monorepo.
# Only simple modifications are allowed (adding source-files to targets, adding simple properties
# like target_include_directories). These modifications will be ported to original
# ya.make files by maintainers. Any complex modifications which can't be ported back to the
# original buildsystem will not be accepted.



add_library(contrib-libs-tensorboard)
target_link_libraries(contrib-libs-tensorboard PUBLIC
  contrib-libs-cxxsupp
  yutil
  contrib-libs-protobuf
)
target_proto_messages(contrib-libs-tensorboard PRIVATE
  ${CMAKE_SOURCE_DIR}/contrib/libs/tensorboard/event.proto
  ${CMAKE_SOURCE_DIR}/contrib/libs/tensorboard/resource_handle.proto
  ${CMAKE_SOURCE_DIR}/contrib/libs/tensorboard/summary.proto
  ${CMAKE_SOURCE_DIR}/contrib/libs/tensorboard/tensor.proto
  ${CMAKE_SOURCE_DIR}/contrib/libs/tensorboard/tensor_shape.proto
  ${CMAKE_SOURCE_DIR}/contrib/libs/tensorboard/types.proto
)
target_proto_addincls(contrib-libs-tensorboard
  ./
  ${CMAKE_SOURCE_DIR}/
  ${CMAKE_BINARY_DIR}
  ${CMAKE_SOURCE_DIR}
  ${CMAKE_SOURCE_DIR}/contrib/libs/protobuf/src
  ${CMAKE_BINARY_DIR}
  ${CMAKE_SOURCE_DIR}/contrib/libs/protobuf/src
)
target_proto_outs(contrib-libs-tensorboard
  --cpp_out=${CMAKE_BINARY_DIR}/
  --cpp_styleguide_out=${CMAKE_BINARY_DIR}/
)
