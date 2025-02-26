# --------------------------------------------------------------------------------------
# Copyright (c) 2014-2022, Nucleic Development Team.
#
# Distributed under the terms of the BSD 3-Clause License.
#
# The full license is in the file LICENSE, distributed with this software.
# --------------------------------------------------------------------------------------
import os
import sys

from setuptools.command.build_ext import build_ext

from .version import __version__, __version_info__


def get_include():
    import os

    return os.path.join(os.path.dirname(__file__), "include")


class CppyBuildExt(build_ext):
    """A custom build extension enforcing c++11 standard on all platforms.

    On Windows, FH4 Exception Handling can be disabled by setting the CPPY_DISABLE_FH4
    environment variable. This avoids requiring VCRUNTIME140_1.dll

    """

    # MSVC does not have a c++11 flag and default to c++14 anyway
    c_opts = {"msvc": ["/EHsc"], "unix": ["-std=c++11"]}

    def build_extensions(self):
        ct = self.compiler.compiler_type
        opts = self.c_opts.get(ct, [])
        cppy_includes = get_include()

        for ext in self.extensions:
            ext.include_dirs.insert(0, cppy_includes)
            ext.extra_compile_args = opts
            if sys.platform == "darwin":
                # Only Unix compilers and their ports have `compiler_so` so on MacOS
                # we can sure it will be present.
                compiler_cmd = self.compiler.compiler_so
                # Check if we are using Clang, accounting for absolute path
                if compiler_cmd is not None and "clang" in compiler_cmd[0]:
                    # If so ensure we use a recent enough version of the stdlib
                    ext.extra_compile_args.append("-stdlib=libc++")
                    ext.extra_link_args.append("-stdlib=libc++")
            if ct == "msvc":
                # Switch to a static build for runtimes, but use dynamic
                # linking for `VCRUNTIME140.dll`, `VCRUNTIME140_1.dll`, and the UCRT.
                # This avoids requiring specific versions of `MSVCP140.dll`, while
                # keeping shared state with the rest of the Python process/extensions.
                is_debug = hasattr(sys, "gettotalrefcount")  # only present in Python debug build

                # Mixing debug and release code is bad practice under Windows. The problem is that the
                # different versions can depend on different fundamental parts of the C++ runtime library,
                # such as how memory is allocated, structures for things like iterators might be
                # different, extra code could be generated to perform operations (e.g. checked iterators).
                # As a consequence we build as debug if python is built with debug symbols.
                debug_ext = "d" if is_debug else ""
                ext.extra_compile_args.append(f"/MT{debug_ext}")
                ext.extra_link_args.extend(
                    [
                        f"ucrt{debug_ext}.lib",
                        f"vcruntime{debug_ext}.lib",
                        f"/nodefaultlib:libucrt{debug_ext}.lib",
                        f"/nodefaultlib:libvcruntime{debug_ext}.lib",
                    ]
                )
                if os.environ.get("CPPY_DISABLE_FH4"):
                    # Disable FH4 Exception Handling implementation so that we don't
                    # require VCRUNTIME140_1.dll. For more details, see:
                    # https://devblogs.microsoft.com/cppblog/making-cpp-exception-handling-smaller-x64/
                    # https://github.com/joerick/cibuildwheel/issues/423#issuecomment-677763904
                    ext.extra_compile_args.append("/d2FH4-")
        build_ext.build_extensions(self)


__all__ = ["__version__", "__version_info__", "get_include", "CppyBuildExt"]
