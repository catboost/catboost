{% set target_host = 'powerpc64le-linux-gnu' %}

include(default)

[settings]
arch=ppc64le
build_type=Release
compiler=gcc
compiler.libcxx=libstdc++11
compiler.version=12
compiler.cppstd=20
os=Linux
[options]

[buildenv]
PATH=+(path)/usr/{{target_host}}/bin
CONAN_CMAKE_FIND_ROOT_PATH=/usr/{{target_host}}
CONAN_CMAKE_SYSROOT=/usr/{{target_host}}
SYSROOT=/usr/{{target_host}}
CC={{target_host}}-gcc
CXX={{target_host}}-g++
CXXFLAGS=-I/usr/{{target_host}}/include/
CFLAGS=-I/usr/{{target_host}}/include/
CHOST={{target_host}}
AR={{target_host}}-ar
AS={{target_host}}-as
RANLIB={{target_host}}-ranlib
LD={{target_host}}-ld
STRIP={{target_host}}-strip
