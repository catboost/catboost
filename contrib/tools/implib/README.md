[![License](http://img.shields.io/:license-MIT-blue.svg)](https://github.com/yugr/Implib.so/blob/master/LICENSE.txt)
[![Build Status](https://github.com/yugr/Implib.so/actions/workflows/ci.yml/badge.svg)](https://github.com/yugr/Implib.so/actions)
[![Total alerts](https://img.shields.io/lgtm/alerts/g/yugr/Implib.so.svg?logo=lgtm&logoWidth=18)](https://lgtm.com/projects/g/yugr/Implib.so/alerts/)
[![Codecov](https://codecov.io/gh/yugr/Implib.so/branch/master/graph/badge.svg)](https://codecov.io/gh/yugr/Implib.so)

# Motivation

On Linux/Android, if you link against shared library you normally use `-lxyz` compiler option which makes your application depend on `libxyz.so`. This would cause `libxyz.so` to be forcedly loaded at program startup (and its constructors to be executed) even if you never call any of its functions.

If you instead want to delay loading of `libxyz.so` (e.g. its unlikely to be used and you don't want to waste resources on it or [slow down startup time](https://lwn.net/Articles/341309/) or you want to select best platform-specific implementation at runtime), you can remove dependency from `LDFLAGS` and issue `dlopen` call manually. But this would cause `ld` to err because it won't be able to statically resolve symbols which are supposed to come from this shared library. At this point you have only two choices:
* emit normal calls to library functions and suppress link errors from `ld` via `-Wl,-z,nodefs`; this is undesired because you loose ability to detect link errors for other libraries statically
* load necessary function addresses at runtime via `dlsym` and call them via function pointers; this isn't very convenient because you have to keep track which symbols your program uses, properly cast function types and also somehow manage global function pointers

Implib.so provides an easy solution - link your program with a _wrapper_ which
* provides all necessary symbols to make linker happy
* loads wrapped library on first call to any of its functions
* redirects calls to library symbols

Generated wrapper code (often also called "shim" code or "shim" library) is analogous to Windows import libraries which achieve the same functionality for DLLs.

Implib.so can also be used to [reduce API provided by existing shared library](doc/ReduceLibraryInterface.md) or [rename it's exported symbols](doc/RenameLibraryInterface.md). See [this page](doc/CMakeIntegration.md) for info on integrating Implib.so into a CMake project.

Implib.so was originally inspired by Stackoverflow question [Is there an elegant way to avoid dlsym when using dlopen in C?](https://stackoverflow.com/questions/45917816/is-there-an-elegant-way-to-avoid-dlsym-when-using-dlopen-in-c/47221180).

# Usage

A typical use-case would look like this:

```
$ implib-gen.py libxyz.so
```

This will generate code for host platform (presumably x86\_64). For other targets do

```
$ implib-gen.py --target $TARGET libxyz.so
```

where `TARGET` can be any of
  * x86\_64-linux-gnu, x86\_64-none-linux-android
  * i686-linux-gnu, i686-none-linux-android
  * arm-linux-gnueabi, armel-linux-gnueabi, armv7-none-linux-androideabi
  * arm-linux-gnueabihf (ARM hardfp ABI)
  * aarch64-linux-gnu, aarch64-none-linux-android
  * mipsel-linux-gnu
  * mips64el-linux-gnuabi64
  * e2k-linux-gnu
  * powerpc64le-linux-gnu
  * powerpc64-linux-gnu (limited support)
  * riscv64-linux-gnu

Script generates two files: `libxyz.so.tramp.S` and `libxyz.so.init.c` which need to be linked to your application (instead of `-lxyz`):

```
$ gcc myfile1.c myfile2.c ... libxyz.so.tramp.S libxyz.so.init.c ... -ldl
```

Note that you need to link against libdl.so. On ARM in case your app is compiled to Thumb code (which e.g. Ubuntu's `arm-linux-gnueabihf-gcc` does by default) you'll also need to add `-mthumb-interwork`.

Application can then freely call functions from `libxyz.so` _without linking to it_. Library will be loaded (via `dlopen`) on first call to any of its functions. If you want to forcedly resolve all symbols (e.g. if you want to avoid delays further on) you can call `void libxyz_init_all()`.

Above command would perform a _lazy load_ i.e. load library on first call to one of it's symbols. If you want to load it at startup, run

```
$ implib-gen.py --no-lazy-load libxyz.so
```

If you don't want `dlopen` to be called automatically and prefer to load library yourself at program startup, run script as

```
$ implib-gen.py --no-dlopen libxys.so
```

If you do want to load library via `dlopen` but would prefer to call it yourself (e.g. with custom parameters or with modified library name), run script as

```
$ cat mycallback.c
#define _GNU_SOURCE
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>

#ifdef __cplusplus
extern "C"
#endif

// Callback that tries different library names
void *mycallback(const char *lib_name) {
  lib_name = lib_name;  // Please the compiler
  void *h;
  h = dlopen("libxyz.so", RTLD_LAZY);
  if (h)
    return h;
  h = dlopen("libxyz-stub.so", RTLD_LAZY);
  if (h)
    return h;
  fprintf(stderr, "dlopen failed: %s\n", dlerror());
  exit(1);
}

$ implib-gen.py --dlopen-callback=mycallback libxyz.so
```

(callback must have signature `void *(*)(const char *lib_name)` and return handle of loaded library).

Normally symbols are located via `dlsym` function but this can be overriden with custom callback
by using `--dlsym-callback` (which must have signature
`void *(*)(void *handle, const char *sym_name)`).

Finally to force library load and resolution of all symbols, call

    void _LIBNAME_tramp_resolve_all(void);

# Wrapping vtables

By default the tool does not try to wrap vtables exported from the library. This can be enabled via `--vtables` flag:
```
$ implib-gen.py --vtables ...
```

# Overhead

Implib.so overhead on a fast path boils down to
* predictable direct jump to wrapper
* load from trampoline table
* predictable untaken direct branch to initialization code
* predictable indirect jump to real function

This is very similar to normal shlib call:
* predictable direct jump to PLT stub
* load from GOT
* predictable indirect jump to real function

so it should have equivalent performance.

# Limitations

The tool does not transparently support all features of POSIX shared libraries. In particular
* it can not provide wrappers for data symbols (except C++ virtual/RTTI tables)
* it makes first call to wrapped functions asynch signal unsafe (as it will call `dlopen` and library constructors)
* it may change semantics if there are multiple definitions of same symbol in different loaded shared objects (runtime symbol interposition is considered a bad practice though)
* it may change semantics because shared library constructors are delayed until when library is loaded

The tool also lacks the following important features:
* proper support for multi-threading
* symbol versions are not handled at all
* keep fast paths of shims together to reduce I$ pressure
* support for macOS and BSDs (actually BSDs mostly work)

Finally, there are some minor TODOs in code.

# Related work

As mentioned in introduction import libraries are first class citizens on Windows platform:
* [Wikipedia on Windows Import Libraries](https://en.wikipedia.org/wiki/Dynamic-link_library#Import_libraries)
* [MSDN on Linker Support for Delay-Loaded DLLs](https://msdn.microsoft.com/en-us/library/151kt790.aspx)

Delay-loaded libraries were once present on OSX (via `-lazy_lXXX` and `-lazy_library` options).

Lazy loading is supported by Solaris shared libraries but was never implemented in Linux. There have been [some discussions](https://www.sourceware.org/ml/libc-help/2013-02/msg00017.html) in libc-alpha but no patches were posted.

Implib.so-like functionality is used in [OpenGL loading libraries](https://www.khronos.org/opengl/wiki/OpenGL_Loading_Library) e.g. [GLEW](http://glew.sourceforge.net/) via custom project-specific scripts.
