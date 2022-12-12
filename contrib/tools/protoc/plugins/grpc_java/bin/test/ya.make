EXECTEST()


DEPENDS(contrib/tools/protoc/plugins/grpc_java/bin/parsever)
INCLUDE(${ARCADIA_ROOT}/contrib/tools/protoc/plugins/grpc_java/bin/ya.version)
RUN(parsever ${GRPC_JAVA_VERSION})

END()
