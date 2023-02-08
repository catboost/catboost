include(default)

target_host=aarch64-linux-gnu

[settings]
arch=armv8
build_type=Release
compiler=clang
compiler.libcxx=libc++
compiler.version=14
compiler.cppstd=20
os=Linux
[options]

[env]
CONAN_CMAKE_FIND_ROOT_PATH=/usr/$target_host
CONAN_CMAKE_SYSROOT=/usr/$target_host
CC=clang
CXX=clang++
CXXFLAGS="-I/usr/$target_host/include/ -target $target_host"
CFLAGS="-I/usr/$target_host/include/ -target $target_host"
CHOST=$target_host
AR=$target_host-ar
AS=$target_host-as
RANLIB=$target_host-ranlib
LD=$target_host-ld
STRIP=$target_host-strip
