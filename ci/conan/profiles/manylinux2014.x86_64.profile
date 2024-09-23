{% set libc, libc_version = detect_api.detect_libc() %}

include(default)

[settings]
arch=x86_64
build_type=Release
compiler=clang
compiler.libcxx=libstdc++11
compiler.version=16
compiler.cppstd=20
os=Linux
os.libc={{libc}}
os.libc_version={{libc_version}}
[options]

[buildenv]
CC=clang-16
CXX=clang-16
