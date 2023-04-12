include(default)

target_host=aarch64-unknown-linux-gnu

[settings]
arch=armv8
build_type=Release
compiler=gcc
compiler.libcxx=libstdc++11
compiler.version=10
compiler.cppstd=20
os=Linux
[options]

[env]
CROSS_ROOT=/usr/xcc/$target_host
CONAN_CMAKE_FIND_ROOT_PATH=/usr/xcc/$target_host
CONAN_CMAKE_SYSROOT=/usr/xcc/$target_host
SYSROOT=/usr/xcc/$target_host
CC=$target_host-gcc
CXX=$target_host-g++
CXXFLAGS="-I/usr/xcc/$target_host/include/ -mno-outline-atomics"
CFLAGS="-I/usr/xcc/$target_host/include/ -mno-outline-atomics"
CHOST=$target_host
AR=$target_host-ar
AS=$target_host-as
RANLIB=$target_host-ranlib
LD=$target_host-ld
STRIP=$target_host-strip
OBJCOPY=$target_host-objcopy
CPP=$target_host-cpp
FC=$target_host-gfortran
