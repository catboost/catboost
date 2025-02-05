[settings]
arch=x86_64
build_type={{ os.getenv("CMAKE_BUILD_TYPE", "Release") }}
compiler=msvc
compiler.cppstd=20
compiler.runtime=static
compiler.runtime_type={{ os.getenv("CMAKE_BUILD_TYPE", "Release") }}
compiler.version=192
os=Windows
[conf]
tools.cmake.cmaketoolchain:generator=Ninja
tools.microsoft.msbuild:vs_version=17
