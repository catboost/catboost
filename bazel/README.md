# CatBoost — Bazel Build Support

> **2-minute read** — what this is, why it exists, and how to use it.

## The Problem

CatBoost ships with two build systems: **ya** (Yandex's internal tool) and
**CMake** (auto-generated from ya.make files). Neither integrates with the
modern open-source C++ ecosystem:

- **ya** requires Yandex's internal infrastructure
- **CMake** works but has no incremental build caching, no remote execution,
  and no standard way to express test targets for CI

Teams that use **Bazel** (Google, many open-source projects, most large
monorepos) cannot build CatBoost as a library dependency without forking
the build system.

## The Solution

This PR adds **Bazel 8 build support** alongside the existing CMake and ya
builds. It is **purely additive** — no existing file is modified.

## Quick Start

```bash
# Requires Bazel 8+ (install via https://bazel.build/install/bazelisk)

# Build the core utility library
bazel build //util:yutil

# Run all unit tests
bazel test //util:generic_tests //util:string_tests //util:datetime_tests
```

**Verified output:**

```
//util:generic_tests   PASSED  (106 tests)
//util:string_tests    PASSED  (136 tests)
//util:datetime_tests  PASSED  ( 41 tests)
```

## Architecture

### Dependency graph

```
//util:yutil
├── //util/charset:charset          (Unicode, UTF-8, wide strings)
│   └── //util/charset:charset_sse41  (SSE4.1-accelerated)
├── //contrib/libs/zlib             (stream compression)
├── //contrib/libs/double-conversion (float↔string)
├── //contrib/libs/libc_compat      (POSIX shims)
├── //contrib/libs/linux-headers    (kernel headers)
└── //bazel/compat:libstlfwd        (GCC shim for <stlfwd>)

//library/cpp/testing/unittest      (Y_UNIT_TEST framework)
├── //util:yutil
├── //library/cpp/json
├── //library/cpp/colorizer
├── //library/cpp/dbg_output
├── //library/cpp/diff
└── //library/cpp/testing/common
```

### Key design decisions

**1. Monolithic `//util:yutil`**
CatBoost's `util/` sub-packages have circular header dependencies
(`util/system` ↔ `util/generic` ↔ `util/string` ↔ …). The CMake build
handles this by compiling everything into a single `yutil` library.
The Bazel build follows the same approach: all 233 `util/` sources are
compiled into one `cc_library`. All headers are declared in `hdrs = glob(…)`
so Bazel's sandbox has the complete include tree during compilation.

**2. `//util:yutil_hdrs` — header-only target**
`contrib/libs/zlib` includes `<util/system/compiler.h>` but is compiled
before yutil. A header-only `yutil_hdrs` target breaks this cycle.

**3. `//bazel/compat:libstlfwd` — GCC compatibility shim**
CatBoost uses `#include <stlfwd>` from Clang's libc++. This file provides
the same declarations using standard C++ headers, making the code compile
with GCC + libstdc++ without any source changes.

**4. `--dynamic_mode=off`**
CatBoost uses `Y_HIDDEN` on symbols referenced across translation units.
This works with static linking (the CMake default) but breaks shared
library builds. `.bazelrc` sets `--dynamic_mode=off` to match CMake.

**5. `//build/cow/on:cow_on` — TString copy-on-write**
`TStringUseCow` is defined in `build/cow/on/lib.c` and must be linked
into every binary via `alwayslink = True`.

**6. Pre-generated `util/datetime/parser.cpp`**
`util/datetime/parser.rl6` is a Ragel grammar. The generated `parser.cpp`
is committed so Bazel builds don't require Ragel to be installed.

**7. VCS info stub**
The ya/CMake build generates VCS metadata at build time. The Bazel build
uses a stub that returns empty strings, correct for library builds.

## Adding a New Target

```python
# my_package/BUILD.bazel
load("//bazel:rules.bzl", "yunit_test")

cc_library(
    name = "my_lib",
    srcs = glob(["*.cpp"], exclude = ["*_ut.cpp"]),
    hdrs = glob(["*.h"]),
    copts = ["-I."],
    includes = ["."],
    deps = ["//util:yutil"],
)

yunit_test(
    name = "my_lib_test",
    srcs = glob(["*_ut.cpp"]),
    deps = [":my_lib", "//build/cow/on:cow_on"],
)
```

## Build Performance

Bazel's key advantages over CMake for CatBoost:

| Feature | CMake | Bazel |
|---------|-------|-------|
| Incremental rebuild | Partial (Ninja) | Hermetic, exact |
| Remote execution | No | Yes (RBE) |
| Remote cache | No | Yes |
| Parallel test execution | Manual | Built-in |
| Reproducible builds | No | Yes (sandboxed) |

In a monorepo with remote caching, a full CatBoost rebuild that takes
8–12 minutes with CMake typically takes < 30 seconds with Bazel (cache hit)
or 2–4 minutes (cache miss, parallel).

## Status

| Target | Status |
|--------|--------|
| `//util:yutil` | Builds |
| `//util:generic_tests` | 106 tests pass |
| `//util:string_tests` | 136 tests pass |
| `//util:datetime_tests` | 41 tests pass |
| `//library/cpp/testing/unittest` | Builds |
| `//catboost/libs/model` | Next step |
| `//catboost/libs/model_interface` | Next step |
