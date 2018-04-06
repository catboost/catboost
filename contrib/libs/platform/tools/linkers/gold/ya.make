RESOURCES_LIBRARY()



INCLUDE(${ARCADIA_ROOT}/contrib/libs/platform/tools/misc/binutils/binutils.resource)

LDFLAGS_FIXED(-fuse-ld=$BINUTILS_ROOT_RESOURCE_GLOBAL/bin/ld.gold)

END()
