import subprocess
import os
import sys
import re

indent = "    "


def escape_special_symbols(strval):
    retval = ""
    for c in strval:
        if c in ("\\", "\""):
            retval += "\\" + c
        elif ord(c) < 0x20:
            retval += c.encode('unicode_escape').decode('utf-8')
        else:
            retval += c
    return retval


def escape_line_feed(strval):
    return re.sub(r'\\n', r'\\n"\\\n' + indent + '"', strval)


def escaped_define(strkey, strval):
    return "#define " + strkey + " \"" + escape_line_feed(escape_special_symbols(strval)) + "\""


def get_build_info(compiler, flags):
    build_info = "Build info:\n"
    build_info += indent + "Compiler: " + compiler + "\n"
    build_info += indent + "Compiler version: \n" + get_compiler_info(compiler) + "\n"
    build_info += indent + "Compile flags: " + (flags if flags else "no flags info")
    return build_info


def get_compiler_info(compiler):
    compiler_binary = os.path.basename(compiler).lower()
    if len(compiler.split(' ')) > 1 or compiler_binary == "ymake" or compiler_binary == "ymake.exe":
        compiler_ver_out = "Build by wrapper. No useful info here."
    else:
        compiler_ver_cmd = [compiler]
        if compiler_binary not in ("cl", "cl.exe"):
            compiler_ver_cmd.append('--version')
        env = os.environ.copy()
        env['LOCALE'] = 'C'
        compiler_ver_out = (
            subprocess.Popen(compiler_ver_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env)
            .stdout.read()
            .decode('utf-8')
        )
    return "\n".join(
        # fmt: off
        f'{indent * 2}{line.strip()}'
        for line in compiler_ver_out.splitlines()
        if line.strip()
        # fmt: on
    )


def filterflags(flags_str):
    badflgs = {
        '-fdebug-prefix-map',
        '-DARCADIA_ROOT',
        '-DARCADIA_BUILD_ROOT',
        '/DARCADIA_ROOT',
        '/DARCADIA_BUILD_ROOT',
    }

    def flags_iter():
        for flag in flags_str.split():
            if not flag:
                continue
            if flag.split('=', 1)[0] in badflgs:
                continue
            yield flag

    return ' '.join(flags_iter())


def main():
    if len(sys.argv) != 4:
        print("Usage: build_info_gen.py <output file> <CXX compiler> <CXX flags>", file=sys.stderr)
        sys.exit(1)
    cxx_compiler = sys.argv[2]
    cxx_flags = filterflags(sys.argv[3])
    with open(sys.argv[1], 'w') as result:
        print("#pragma once\n", file=result)
        print(escaped_define("BUILD_INFO", get_build_info(cxx_compiler, cxx_flags)), file=result)
        print(escaped_define("BUILD_COMPILER", cxx_compiler), file=result)
        print(escaped_define("BUILD_COMPILER_VERSION", get_compiler_info(cxx_compiler)), file=result)
        print(escaped_define("BUILD_COMPILER_FLAGS", cxx_flags), file=result)


if __name__ == "__main__":
    main()
