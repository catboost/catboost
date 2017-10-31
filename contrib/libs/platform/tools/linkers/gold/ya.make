LIBRARY()



NO_PLATFORM()

PEERDIR(contrib/libs/platform/tools/misc/binutils)

LDFLAGS("-fuse-ld=$(BINUTILS_ROOT)/bin/ld.gold")

END()
