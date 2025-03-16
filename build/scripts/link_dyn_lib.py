from __future__ import print_function

import sys
import os
import json
import subprocess
import tempfile
import collections
import optparse

try:
    import shlex
    shlex_join = shlex.join
except AttributeError:
    import pipes

    def shlex_join(cmd):
        # equivalent to shlex.join() in python 3
        return ' '.join(pipes.quote(part) for part in cmd)

# Explicitly enable local imports
# Don't forget to add imported scripts to inputs of the calling command!
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import thinlto_cache
import link_exe

from process_whole_archive_option import ProcessWholeArchiveOption



def parse_export_file(p):
    with open(p, 'r') as f:
        for l in f:
            l = l.strip()

            if l and '#' not in l:
                words = l.split()
                if len(words) == 2 and words[0] == 'linux_version':
                    yield {'linux_version': words[1]}
                elif len(words) == 2:
                    yield {'lang': words[0], 'sym': words[1]}
                elif len(words) == 1:
                    yield {'lang': 'C', 'sym': words[0]}
                else:
                    raise Exception('unsupported exports line: ' + l)


def to_c(sym):
    symbols = collections.deque(sym.split('::'))
    c_prefixes = [  # demangle prefixes for c++ symbols
        '_ZN',  # namespace
        '_ZTIN',  # typeinfo for
        '_ZTSN',  # typeinfo name for
        '_ZTTN',  # VTT for
        '_ZTVN',  # vtable for
        '_ZNK',  # const methods
    ]
    c_sym = ''
    while symbols:
        s = symbols.popleft()
        if s == '*':
            c_sym += '*'
            break
        if '*' in s and len(s) > 1:
            raise Exception('Unsupported format, cannot guess length of symbol: ' + s)
        c_sym += str(len(s)) + s
    if symbols:
        raise Exception('Unsupported format: ' + sym)
    if c_sym[-1] != '*':
        c_sym += 'E*'
    return ['{prefix}{sym}'.format(prefix=prefix, sym=c_sym) for prefix in c_prefixes]


def fix_darwin_param(ex):
    for item in ex:
        if item.get('linux_version'):
            continue

        if item['lang'] == 'C':
            yield '-Wl,-exported_symbol,_' + item['sym']
        elif item['lang'] == 'C++':
            for sym in to_c(item['sym']):
                yield '-Wl,-exported_symbol,_' + sym
        else:
            raise Exception('unsupported lang: ' + item['lang'])


def fix_gnu_param(arch, ex):
    d = collections.defaultdict(list)
    version = None
    for item in ex:
        if item.get('linux_version'):
            if not version:
                version = item.get('linux_version')
            else:
                raise Exception('More than one linux_version defined')
        elif item['lang'] == 'C++':
            d['C'].extend(to_c(item['sym']))
        else:
            d[item['lang']].append(item['sym'])

    with tempfile.NamedTemporaryFile(mode='wt', delete=False) as f:
        if version:
            f.write('{} {{\nglobal:\n'.format(version))
        else:
            f.write('{\nglobal:\n')

        for k, v in d.items():
            f.write('    extern "' + k + '" {\n')

            for x in v:
                f.write('        ' + x + ';\n')

            f.write('    };\n')

        f.write('local: *;\n};\n')

        ret = ['-Wl,--version-script=' + f.name]

        if arch == 'ANDROID':
            ret += ['-Wl,--export-dynamic']

        return ret


def fix_windows_param(ex):
    with tempfile.NamedTemporaryFile(delete=False) as def_file:
        exports = []
        for item in ex:
            if item.get('lang') == 'C':
                exports.append(item.get('sym'))
        def_file.write('EXPORTS\n')
        for export in exports:
            def_file.write('    {}\n'.format(export))
        return ['/DEF:{}'.format(def_file.name)]


def fix_cmd(arch, c):
    if arch == 'WINDOWS':
        prefix = '/DEF:'
        f = fix_windows_param
    else:
        prefix = '-Wl,--version-script='
        if arch in ('DARWIN', 'IOS', 'IOSSIM'):
            f = fix_darwin_param
        else:
            f = lambda x: fix_gnu_param(arch, x)

    def do_fix(p):
        if p.startswith(prefix) and p.endswith('.exports'):
            fname = p[len(prefix) :]

            return list(f(list(parse_export_file(fname))))

        if p.endswith('.pkg.fake'):
            return []

        return [p]

    return sum((do_fix(x) for x in c), [])


def parse_args(args):
    parser = optparse.OptionParser()
    parser.disable_interspersed_args()
    parser.add_option('--arch')
    parser.add_option('--target')
    parser.add_option('--soname')
    parser.add_option('--source-root')
    parser.add_option('--build-root')
    parser.add_option('--fix-elf')
    parser.add_option('--linker-output')
    parser.add_option('--dynamic-cuda', action='store_true')
    parser.add_option('--objcopy-exe')
    parser.add_option('--whole-archive-peers', action='append')
    parser.add_option('--whole-archive-libs', action='append')
    parser.add_option('--custom-step')
    parser.add_option('--python')
    thinlto_cache.add_options(parser)
    return parser.parse_args(args)


if __name__ == '__main__':
    args = sys.argv[1:]
    plugins = []

    if '--start-plugins' in args:
        ib = args.index('--start-plugins')
        ie = args.index('--end-plugins')
        plugins = args[ib + 1:ie]
        args = args[:ib] + args[ie + 1:]

    for p in plugins:
        res = subprocess.check_output([sys.executable, p] + args).decode().strip()

        if res:
            args = json.loads(res)

    opts, args = parse_args(args)

    assert opts.arch
    assert opts.target

    cmd = args
    cmd = fix_cmd(opts.arch, cmd)

    cmd = ProcessWholeArchiveOption(opts.arch, opts.whole_archive_peers, opts.whole_archive_libs).construct_cmd(cmd)
    thinlto_cache.preprocess(opts, cmd)

    if opts.custom_step:
        assert opts.python
        subprocess.check_call([opts.python] + [opts.custom_step] + cmd)

    if opts.linker_output:
        stdout = open(opts.linker_output, 'w')
    else:
        stdout = sys.stdout

    proc = subprocess.Popen(cmd, shell=False, stderr=sys.stderr, stdout=stdout)
    proc.communicate()
    thinlto_cache.postprocess(opts)

    if proc.returncode:
        print('linker has failed with retcode:', proc.returncode, file=sys.stderr)
        print('linker command:', shlex_join(cmd), file=sys.stderr)
        sys.exit(proc.returncode)

    if opts.fix_elf:
        cmd = [opts.fix_elf, opts.target]
        proc = subprocess.Popen(cmd, shell=False, stderr=sys.stderr, stdout=sys.stdout)
        proc.communicate()

        if proc.returncode:
            print('fix_elf has failed with retcode:', proc.returncode, file=sys.stderr)
            print('fix_elf command:', shlex_join(cmd), file=sys.stderr)
            sys.exit(proc.returncode)

    if opts.soname and opts.soname != opts.target:
        if os.path.exists(opts.soname):
            os.unlink(opts.soname)
        os.link(opts.target, opts.soname)


# -----------------Test---------------- #
def write_temp_file(content):
    import yatest.common as yc

    filename = yc.output_path('test.exports')
    with open(filename, 'w') as f:
        f.write(content)
    return filename


def test_fix_cmd_darwin():
    export_file_content = """
C++ geobase5::details::lookup_impl::*
C++ geobase5::hardcoded_service
"""
    filename = write_temp_file(export_file_content)
    args = ['-Wl,--version-script={}'.format(filename)]
    assert fix_cmd('DARWIN', args) == [
        '-Wl,-exported_symbol,__ZN8geobase57details11lookup_impl*',
        '-Wl,-exported_symbol,__ZTIN8geobase57details11lookup_impl*',
        '-Wl,-exported_symbol,__ZTSN8geobase57details11lookup_impl*',
        '-Wl,-exported_symbol,__ZTTN8geobase57details11lookup_impl*',
        '-Wl,-exported_symbol,__ZTVN8geobase57details11lookup_impl*',
        '-Wl,-exported_symbol,__ZNK8geobase57details11lookup_impl*',
        '-Wl,-exported_symbol,__ZN8geobase517hardcoded_serviceE*',
        '-Wl,-exported_symbol,__ZTIN8geobase517hardcoded_serviceE*',
        '-Wl,-exported_symbol,__ZTSN8geobase517hardcoded_serviceE*',
        '-Wl,-exported_symbol,__ZTTN8geobase517hardcoded_serviceE*',
        '-Wl,-exported_symbol,__ZTVN8geobase517hardcoded_serviceE*',
        '-Wl,-exported_symbol,__ZNK8geobase517hardcoded_serviceE*',
    ]


def run_fix_gnu_param(export_file_content):
    filename = write_temp_file(export_file_content)
    result = fix_gnu_param('LINUX', list(parse_export_file(filename)))[0]
    version_script_path = result[len('-Wl,--version-script=') :]
    with open(version_script_path) as f:
        content = f.read()
    return content


def test_fix_gnu_param():
    export_file_content = """
C++ geobase5::details::lookup_impl::*
C   getFactoryMap
"""
    assert (
        run_fix_gnu_param(export_file_content)
        == """{
global:
    extern "C" {
        _ZN8geobase57details11lookup_impl*;
        _ZTIN8geobase57details11lookup_impl*;
        _ZTSN8geobase57details11lookup_impl*;
        _ZTTN8geobase57details11lookup_impl*;
        _ZTVN8geobase57details11lookup_impl*;
        _ZNK8geobase57details11lookup_impl*;
        getFactoryMap;
    };
local: *;
};
"""
    )


def test_fix_gnu_param_with_linux_version():
    export_file_content = """
C++ geobase5::details::lookup_impl::*
linux_version ver1.0
C   getFactoryMap
"""
    assert (
        run_fix_gnu_param(export_file_content)
        == """ver1.0 {
global:
    extern "C" {
        _ZN8geobase57details11lookup_impl*;
        _ZTIN8geobase57details11lookup_impl*;
        _ZTSN8geobase57details11lookup_impl*;
        _ZTTN8geobase57details11lookup_impl*;
        _ZTVN8geobase57details11lookup_impl*;
        _ZNK8geobase57details11lookup_impl*;
        getFactoryMap;
    };
local: *;
};
"""
    )
