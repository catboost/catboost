OS_TYPE := $(shell uname)

.PHONY : all

ifeq ($(OS_TYPE),Linux)
all: libcatboostr.so
else
all: libcatboostr.dylib
endif

libcatboostr.dylib libcatboostr.so:
	$(MAKE) -f Makefile.inner

clean:
ifeq ($(OSTYPE), Linux)
	rm -rf libcatboostr.so
else
	rm -rf libcatboostr.dylib
endif
