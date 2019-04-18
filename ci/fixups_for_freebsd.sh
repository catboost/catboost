#!/usr/bin/env bash

set -e

# fixup platform
sed -i -e 's/x86_64\-linux\-gnu/x86_64\-unknown\-freebsd11\.2/' make/*.makefile
# use clang 6.0, not 5.0
sed -i -e "s/\'5 0\'/\'6 0\'/" make/*.makefile
# don't build AFALG on freebsd
echo '##define OPENSSL_NO_AFALGENG' >> contrib/libs/openssl/1.1.1/include/openssl/opensslconf.h
# hack
echo '##include <stdio.h>' > contrib/libs/openssl/1.1.1/engines/e_afalg.c
