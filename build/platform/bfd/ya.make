RESOURCES_LIBRARY()



INCLUDE(${ARCADIA_ROOT}/build/platform/binutils/binutils.resource)

LDFLAGS(-fuse-ld=$BINUTILS_ROOT_RESOURCE_GLOBAL/bin/ld.bfd)

END()
