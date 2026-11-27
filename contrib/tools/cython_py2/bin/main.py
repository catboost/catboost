def main():
    import argparse
    import sys

    from Cython.Compiler import Options
    from Cython.Compiler.Main import main as compile_main

    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument("--target-platform")
    settings, arguments = parser.parse_known_args()
    platform = {"LINUX": "Linux", "DARWIN": "Darwin", "WINDOWS": "Windows"}.get(settings.target_platform)
    if platform:
        # Target settings precede explicit compiler options, so callers can override them.
        arguments = ["-E", "UNAME_SYSNAME=" + platform] + arguments
    sys.argv[1:] = arguments

    # Keep Arcadia exception semantics; explicit CLI directives still override this default.
    Options.get_directive_defaults()["legacy_implicit_noexcept"] = True
    compile_main(command_line=1)
