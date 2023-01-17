PY3_PROGRAM()



PEERDIR(library/python/resource)

RESOURCE(
    ${ARCADIA_ROOT}/contrib/libs/grpc-java/core/src/main/java/io/grpc/internal/GrpcUtil.java /GrpcUtil.java
)

PY_SRCS(__main__.py)

END()
