# CatBoost — Bazel Build Support

This directory contains the Bazel build configuration for CatBoost.
The Bazel build is **additive** — no existing CMake or `ya` build files
are modified.

## Requirements

| Tool | Version |
|------|---------|
| Bazel | 8.1+ (Bzlmod) |
| GCC | 11+ or Clang 14+ |
| C++ standard | C++20 |

Install Bazel via [Bazelisk](https://github.com/bazelbuild/bazelisk):
```bash
go install github.com/bazelbuild/bazelisk@latest
# or
curl -Lo bazel https://github.com/bazelbuild/bazelisk/releases/latest/download/bazelisk-linux-amd64
chmod +x bazel && sudo mv bazel /usr/local/bin/bazel
```

## Quick Start

```bash
# Build the core utility library (yutil)
bazel build //util:yutil

# Build the CatBoost model library
bazel build //catboost/libs/model:catboost_model

# Build the CatBoost C API
bazel build //catboost/libs/model_interface:catboost_capi
```

## Architecture

The Bazel build mirrors the CMake dependency graph:

```
//catboost/libs/model:catboost_model
    └── //util:yutil
            ├── //util/charset:charset_generated
            ├── //contrib/libs/zlib:zlib
            ├── //contrib/libs/double-conversion:double-conversion
            ├── //contrib/libs/libc_compat:libc_compat
            └── //contrib/libs/cxxsupp:cxxsupp  (GCC stlfwd shim)
```

## Key Design Decisions

### 1. Monolithic `yutil` target
CatBoost's `util/` library has circular header dependencies between
sub-packages (e.g., `util/system` ↔ `util/generic`). The CMake build
handles this by compiling everything into a single `yutil` library.
The Bazel build follows the same approach: all `util/` sources are
compiled into `//util:yutil`.

### 2. `includes = ["."]` per package
Each `cc_library` uses `includes = ["."]` to add its own directory to
the system include path. This makes relative includes like
`#include "platform.h"` work correctly within each package.

### 3. Workspace root on include path
The `.bazelrc` adds `-I.` to all compilations so that
`#include <util/generic/string.h>` resolves from the workspace root.

### 4. GCC compatibility shim (`bazel/compat/stlfwd`)
CatBoost uses `#include <stlfwd>` from the Clang-specific libcxx.
The `bazel/compat/stlfwd` file provides a GCC/libstdc++ compatible
replacement that includes the standard forward-declaration headers.

### 5. Bzlmod (`MODULE.bazel`)
Uses Bazel's modern dependency management (Bzlmod) instead of the
legacy `WORKSPACE` file. All external dependencies are fetched from
the [Bazel Central Registry](https://registry.bazel.build/).

## Adding New Targets

To add a new CatBoost library:

```python
# catboost/libs/my_lib/BUILD.bazel
cc_library(
    name = "my_lib",
    srcs = glob(["*.cpp"], exclude = ["*_ut.cpp"]),
    hdrs = glob(["*.h"]),
    includes = ["."],
    deps = ["//util:yutil"],
)
```

## Status

| Target | Status |
|--------|--------|
| `//util:yutil` | ✅ Builds |
| `//contrib/libs/zlib:zlib` | ✅ Builds |
| `//contrib/libs/double-conversion:double-conversion` | ✅ Builds |
| `//contrib/libs/libc_compat:libc_compat` | ✅ Builds |
| `//catboost/libs/model:catboost_model` | 🚧 In progress |
| Tests | 🚧 Requires `library/cpp/testing/unittest` BUILD files |
