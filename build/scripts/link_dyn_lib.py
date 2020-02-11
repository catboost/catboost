import sys
import os
import subprocess
import tempfile
import collections
import optparse
import pipes


def shlex_join(cmd):
    # equivalent to shlex.join() in python 3
    return ' '.join(
        pipes.quote(part)
        for part in cmd
    )


def parse_export_file(p):
    with open(p, 'r') as f:
        for l in f.read().split('\n'):
            l = l.strip()

            if l and '#' not in l:
                x, y = l.split()
                if x == 'linux_version':
                    yield {'linux_version': y}
                else:
                    yield {'lang': x, 'sym': y}


def to_c(sym):
    symbols = collections.deque(sym.split('::'))
    c_prefixes = [  # demangle prefixes for c++ symbols
        '_ZN',      # namespace
        '_ZTIN',    # typeinfo for
        '_ZTSN',    # typeinfo name for
        '_ZTTN',    # VTT for
        '_ZTVN',    # vtable for
        '_ZNK',     # const methods
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


musl_libs = '-lc', '-lcrypt', '-ldl', '-lm', '-lpthread', '-lrt', '-lutil'


def fix_cmd(arch, musl, c):
    if arch == 'WINDOWS':
        prefix = '/DEF:'
        f = fix_windows_param
    else:
        prefix = '-Wl,--version-script='
        if arch in ('DARWIN', 'IOS'):
            f = fix_darwin_param
        else:
            f = lambda x: fix_gnu_param(arch, x)

    def do_fix(p):
        if musl and p in musl_libs:
            return []

        if p.startswith(prefix) and p.endswith('.exports'):
            fname = p[len(prefix):]

            return list(f(list(parse_export_file(fname))))

        if p.endswith('.supp.o'):
            return []

        return [p]

    return sum((do_fix(x) for x in c), [])


def postprocess_whole_archive(opts, args):
    if not opts.whole_archive:
        return args

    def match_peer_lib(arg, peers):
        key = None
        if arg.endswith('.a'):
            key = os.path.dirname(arg)
        return key if key and key in peers else None

    whole_archive_peers = { x : 0 for x in opts.whole_archive }

    whole_archive_flag = '-Wl,--whole-archive'
    no_whole_archive_flag = '-Wl,--no-whole-archive'

    cmd = []
    is_inside_whole_archive = False
    is_whole_archive = False
    # We are trying not to create excessive sequences of consecutive flags
    # -Wl,--no-whole-archive  -Wl,--whole-archive ('externally' specified
    # flags -Wl,--[no-]whole-archive are not taken for consideration in this
    # optimization intentionally)
    for arg in args:
        if arg == whole_archive_flag:
            is_inside_whole_archive = True
            is_whole_archive = False
        elif arg == no_whole_archive_flag:
            is_inside_whole_archive = False
            is_whole_archive = False
        else:
            key = match_peer_lib(arg, whole_archive_peers)
            if key:
                whole_archive_peers[key] += 1

            if not is_inside_whole_archive:
                if key:
                    if not is_whole_archive:
                        cmd.append(whole_archive_flag)
                        is_whole_archive = True
                elif is_whole_archive:
                    cmd.append(no_whole_archive_flag)
                    is_whole_archive = False

        cmd.append(arg)

    if is_whole_archive:
        cmd.append(no_whole_archive_flag)

    for key, value in whole_archive_peers.items():
        assert value > 0, '"{}" specified in WHOLE_ARCHIVE() macro is not used on link command'.format(key)

    return cmd


def parse_args():
    parser = optparse.OptionParser()
    parser.disable_interspersed_args()
    parser.add_option('--arch')
    parser.add_option('--target')
    parser.add_option('--soname')
    parser.add_option('--fix-elf')
    parser.add_option('--musl', action='store_true')
    parser.add_option('--whole-archive', action='append')
    return parser.parse_args()


if __name__ == '__main__':
    opts, args = parse_args()

    assert opts.arch
    assert opts.target

    cmd = fix_cmd(opts.arch, opts.musl, args)
    cmd = postprocess_whole_archive(opts, cmd)
    proc = subprocess.Popen(cmd, shell=False, stderr=sys.stderr, stdout=sys.stdout)
    proc.communicate()

    if proc.returncode:
        print >>sys.stderr, 'linker has failed with retcode:', proc.returncode
        print >>sys.stderr, 'linker command:', shlex_join(cmd)
        sys.exit(proc.returncode)

    if opts.fix_elf:
        cmd = [opts.fix_elf, opts.target]
        proc = subprocess.Popen(cmd, shell=False, stderr=sys.stderr, stdout=sys.stdout)
        proc.communicate()

        if proc.returncode:
            print >>sys.stderr, 'fix_elf has failed with retcode:', proc.returncode
            print >>sys.stderr, 'fix_elf command:', shlex_join(cmd)
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
    assert fix_cmd('DARWIN', False, args) == [
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
    version_script_path = result[len('-Wl,--version-script='):]
    with open(version_script_path) as f:
        content = f.read()
    return content


def test_fix_gnu_param():
    export_file_content = """
C++ geobase5::details::lookup_impl::*
C   getFactoryMap
"""
    assert run_fix_gnu_param(export_file_content) == """{
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


def test_fix_gnu_param_with_linux_version():
    export_file_content = """
C++ geobase5::details::lookup_impl::*
linux_version ver1.0
C   getFactoryMap
"""
    assert run_fix_gnu_param(export_file_content) == """ver1.0 {
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
