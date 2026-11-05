# Little CMS - Build Instructions

This document describes the supported build systems for Little CMS (lcms2),
their status, and general guidance for building and installing the library.

---

## Build Systems Overview

Little CMS supports three build systems:

| Build System | Status |
|---|---|
| Autotools (autoconf/automake/libtool) | **Fully Supported** |
| CMake | **Supported, in active development** |
| Meson | **Supported, in testing** |

Autotools remains the traditional and well-tested build system for Unix-like
platforms. CMake has been added to support native Windows toolchains and
environments where Autotools is unavailable or impractical. Meson support is
being evaluated.

> The addition of CMake does **not** signal an intent to remove Autotools.
> Both are supported in parallel. Users are encouraged to report issues with
> any of the build systems.

---

## Requirements for All Build Systems

A supported build system must satisfy the following requirements:

- Build the library and all utilities correctly.
- Support installation to alternate prefixes (`--prefix` or equivalent).
- Support cross-compilation.
- Provide a `make test` equivalent that uses the **just-built** library,
  not the installed one (RPATH or equivalent care required).
- Run correctly on GNU/Linux, macOS, *BSD, and Cygwin/MSYS2
- Install CMake package config files (for use by downstream CMake projects),
  regardless of which build system was used to build lcms2.
- Support creation of distribution tarballs that match the canonical release
  content (not necessarily identical to the raw GitHub archive).

---

## Building with Autotools

**Prerequisites:** `autoconf`, `automake`, `libtool`

    ./autogen.sh        # if building from a git checkout
    ./configure
    make
    make check
    make install

To install to a custom prefix:

    ./configure --prefix=/your/prefix

**Notes:**
- Autotools builds are the reference for test coverage and correctness.
- Autotools does **not** support native Windows builds in environments without
  a POSIX shell (e.g. MSVC-only environments). Use CMake in that case.

---

## Building with CMake

**Prerequisites:** CMake 3.x or later

    cmake -S . -B build
    cmake --build build
    ctest --test-dir build
    cmake --install build

To install to a custom prefix:

    cmake -S . -B build -DCMAKE_INSTALL_PREFIX=/your/prefix

**Notes:**
- CMake is the recommended build system for native Windows (MSVC) builds.
- CMake package config files (`lcms2Config.cmake`, etc.) are installed and
  allow other CMake-based projects to find and link against lcms2.
  These config files should be available regardless of how lcms2 was built.
- Cross-compilation is supported via standard CMake toolchain files.

---

## Building with Meson

**Prerequisites:** `meson`, `ninja`

    meson setup build
    ninja -C build
    meson test -C build
    meson install -C build

To install to a custom prefix:

    meson setup build --prefix=/your/prefix

**Notes:**
- Meson support is currently in testing. Please report any issues.

---

## Distribution Tarballs

> Always prefer the official release tarballs published on the Little CMS
> website or SourceForge over raw GitHub archives when packaging for
> distributions.

---

## Reporting Build Issues

Please report build system issues to the lcms-user mailing list:
- **Mailing list:** lcms-user@lists.sourceforge.net
- **GitHub Issues:** https://github.com/mm2/Little-CMS/issues

When reporting, please include:
- The build system used (Autotools / CMake / Meson) and its version.
- The operating system and architecture.
- Whether you are cross-compiling and, if so, the host/target tuple.
- The full error output.

---

## Further Information

| | |
|---|---|
| Website | https://www.littlecms.com |
| Repository | https://github.com/mm2/Little-CMS |
| Mailing list | lcms-user@lists.sourceforge.net |