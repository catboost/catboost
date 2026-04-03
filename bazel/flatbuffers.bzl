"""Bazel macro for generating C++ code from FlatBuffers schema files (.fbs).

Usage:
    flatbuffers_library(
        name = "my_fbs",
        srcs = ["my_schema.fbs"],
        include_paths = ["path/to/includes"],
    )
"""

def flatbuffers_library(name, srcs, include_paths = [], deps = [], visibility = None):
    """Generate C++ headers and sources from FlatBuffers .fbs schema files.

    Args:
        name:          Bazel target name.
        srcs:          List of .fbs schema files.
        include_paths: Additional include paths passed to flatc via -I.
        deps:          Additional flatbuffers_library deps (for schema imports).
        visibility:    Bazel visibility.
    """
    outs_h = []
    outs_cpp = []

    for src in srcs:
        # Derive base name without extension
        base = src.replace(".fbs", "")
        outs_h.append(base + ".fbs.h")
        outs_cpp.append(base + ".fbs.cpp")

    include_flags = " ".join(["-I " + p for p in include_paths])

    native.genrule(
        name = name + "_gen",
        srcs = srcs,
        outs = outs_h + outs_cpp,
        cmd = """
            FLATC=$(location //contrib/libs/flatbuffers/flatc)
            for fbs in $(SRCS); do
                $$FLATC --cpp --gen-object-api --reflect-names {include_flags} \\
                    --filename-suffix ".fbs" \\
                    -o $(@D) $$fbs
            done
        """.format(include_flags = include_flags),
        tools = ["//contrib/libs/flatbuffers/flatc"],
    )

    native.cc_library(
        name = name,
        srcs = outs_cpp,
        hdrs = outs_h,
        deps = deps + [
            "//contrib/libs/flatbuffers",
        ],
        visibility = visibility,
    )
