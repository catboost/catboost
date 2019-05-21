#!/usr/bin/env bash

set -e

# fixup platform
sed -i -e 's/x86_64\-linux\-gnu/x86_64\-unknown\-freebsd11\.2/' make/*.makefile
# use clang 6.0, not 5.0
sed -i -e "s/\'5 0\'/\'6 0\'/" make/*.makefile
# endian.h => sys/endian.h
sed -i -e "s/<endian.h>/<sys\/endian.h>/" contrib/python/numpy/numpy/core/include/numpy/npy_endian.h
# don't build AFALG on freebsd
echo '#define OPENSSL_NO_AFALGENG' >> contrib/libs/openssl/include/openssl/opensslconf.h
# hack
echo '#include <stdio.h>' > contrib/libs/openssl/engines/e_afalg.c
