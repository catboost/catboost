"""Bazel macro for generating enum serialization code via the enum_parser tool.

The enum_parser tool reads C++ header files, extracts enum definitions, and
generates a .cpp file containing ToString / FromString / Out overloads.

Usage:
    generate_enum_serialization(
        name      = "my_lib_enum_serialization",
        src       = "my_header.h",
        includes  = ["my/package/my_header.h"],
        extra_headers = ["other_needed_header.h"],
    )
"""

def generate_enum_serialization(name, src, includes = [], extra_headers = [], visibility = None):
    """Generate enum serialization .cpp source from a C++ header.

    Args:
        name:          Bazel target name for the generated cc_library.
        src:           The header file label containing enums to serialize.
        includes:      List of include paths to pass to the enum parser
                       (workspace-relative paths used in the generated #include
                       directives).
        extra_headers: Additional header file labels that the generated file
                       should include.
        visibility:    Bazel visibility.
    """
    out_cpp = name + ".cpp"

    include_flags = " ".join(["--include-path " + p for p in includes])

    native.genrule(
        name = name + "_gen",
        srcs = [src] + extra_headers,
        outs = [out_cpp],
        cmd = """
            $(location //tools/enum_parser/enum_parser) \\
                $(location {src}) \\
                {include_flags} \\
                --output-file $@
        """.format(src = src, include_flags = include_flags),
        tools = ["//tools/enum_parser/enum_parser"],
    )

    native.cc_library(
        name = name,
        srcs = [out_cpp],
        deps = [
            "//tools/enum_parser/enum_serialization_runtime",
            "//util",
        ],
        visibility = visibility or ["//visibility:public"],
    )
