#!/usr/bin/env python
# coding=utf-8

import base64
import itertools
import json
import logging
import ntpath
import optparse
import os
import posixpath
import re
import subprocess
import sys
import tempfile

logger = logging.getLogger(__name__ if __name__ != '__main__' else 'ymake_conf.py')


def init_logger(verbose):
    logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO)


class DebugString(object):
    def __init__(self, get_string_func):
        self.get_string_func = get_string_func

    def __str__(self):
        return self.get_string_func()


class ConfigureError(Exception):
    pass


class Platform(object):
    def __init__(self, name, os, arch):
        """
        :type name: str
        :type os: str
        :type arch: str
        """
        self.name = name
        self.os = self._parse_os(os)
        self.arch = arch.lower()

        self.is_i386 = self.arch in ('i386', 'x86')
        self.is_i686 = self.arch == 'i686'
        self.is_x86 = self.is_i386 or self.is_i686
        self.is_x86_64 = self.arch in ('x86_64', 'amd64')
        self.is_intel = self.is_x86 or self.is_x86_64

        self.is_armv7 = self.arch in ('armv7', 'armv7a', 'armv7a_neon', 'arm', 'armv7ahf_cortex_a35', 'armv7ahf_cortex_a53')
        self.is_armv8 = self.arch in ('armv8', 'armv8a', 'arm64', 'aarch64', 'armv8a_cortex_a35', 'armv8a_cortex_a53')
        self.is_arm = self.is_armv7 or self.is_armv8
        self.is_armv7_neon = self.arch in ('armv7a_neon', 'armv7ahf_cortex_a35', 'armv7ahf_cortex_a53')

        self.is_cortex_a35 = self.arch in ('armv7ahf_cortex_a35', 'armv8a_cortex_a35')
        self.is_cortex_a53 = self.arch in ('armv7ahf_cortex_a53', 'armv8a_cortex_a53')

        self.is_armv7hf = self.arch in ('armv7ahf_cortex_a35', 'armv7ahf_cortex_a53')

        self.is_ppc64le = self.arch == 'ppc64le'

        self.is_32_bit = self.is_x86 or self.is_armv7
        self.is_64_bit = self.is_x86_64 or self.is_armv8 or self.is_ppc64le

        assert self.is_32_bit or self.is_64_bit
        assert not (self.is_32_bit and self.is_64_bit)

        self.is_linux = self.os == 'linux' or 'yocto' in self.os
        self.is_linux_x86_64 = self.is_linux and self.is_x86_64
        self.is_linux_armv8 = self.is_linux and self.is_armv8
        self.is_linux_armv7 = self.is_linux and self.is_armv7

        self.is_macos = self.os == 'macos'
        self.is_macos_x86_64 = self.is_macos and self.is_x86_64
        self.is_ios = self.os == 'ios'
        self.is_apple = self.is_macos or self.is_ios

        self.is_windows = self.os == 'windows'
        self.is_windows_x86_64 = self.is_windows and self.is_x86_64

        self.is_android = self.os == 'android'
        self.is_cygwin = self.os == 'cygwin'
        self.is_freebsd = self.os == 'freebsd'
        self.is_yocto = self.os == 'yocto'
        self.is_yocto_lg_wk7y = self.os == 'yocto_lg_wk7y' and self.is_armv7
        self.is_yocto_jbl_portable_music = self.os == 'yocto_jbl_portable_music' and self.is_armv7
        self.is_yocto_aacrh64_lightcomm_mt8516 = self.os == 'yocto_lightcomm' and self.is_armv8

        self.is_posix = self.is_linux or self.is_apple or self.is_android or self.is_cygwin or self.is_freebsd or self.is_yocto or self.is_yocto_lg_wk7y or self.is_yocto_jbl_portable_music or self.is_yocto_aacrh64_lightcomm_mt8516

    @staticmethod
    def from_json(data):
        name = data.get('visible_name', data['toolchain'])
        return Platform(name, os=data['os'], arch=data['arch'])

    @property
    def os_variables(self):
        # 'LINUX' variable, for backward compatibility
        yield self.os.upper()

        # 'OS_LINUX' variable
        yield 'OS_{}'.format(self.os.upper())

        # yocto is linux
        if 'yocto' in self.os:
            yield 'LINUX'
            yield 'OS_LINUX'

        if self.is_macos:
            yield 'DARWIN'
            yield 'OS_DARWIN'

    @property
    def arch_variables(self):
        return select_multiple((
            (self.is_i386 or self.is_i686, 'ARCH_I386'),
            (self.is_i686, 'ARCH_I686'),
            (self.is_x86_64, 'ARCH_X86_64'),
            (self.is_armv7, 'ARCH_ARM7'),
            (self.is_armv7_neon, 'ARCH_ARM7_NEON'),
            (self.is_armv8, 'ARCH_ARM64'),
            (self.is_arm, 'ARCH_ARM'),
            (self.is_linux_armv8, 'ARCH_AARCH64'),
            (self.is_ppc64le, 'ARCH_PPC64LE'),
            (self.is_32_bit, 'ARCH_TYPE_32'),
            (self.is_64_bit, 'ARCH_TYPE_64'),
        ))

    @property
    def library_path_variables(self):
        return ['LD_LIBRARY_PATH', 'DYLD_LIBRARY_PATH']

    def find_in_dict(self, dict_, default=None):
        if dict_ is None:
            return default
        for key in dict_.iterkeys():
            if self._parse_os(key) == self.os:
                return dict_[key]
        return default

    @property
    def os_compat(self):
        if self.is_macos:
            return 'DARWIN'
        else:
            return self.os.upper()

    def exe(self, *paths):
        if self.is_windows:
            return ntpath.join(*itertools.chain(paths[:-1], (paths[-1] + '.exe',)))
        else:
            return posixpath.join(*paths)

    def __str__(self):
        return '{name}-{os}-{arch}'.format(name=self.name, os=self.os, arch=self.arch)

    def __cmp__(self, other):
        return cmp((self.name, self.os, self.arch), (other.name, other.os, other.arch))

    def __hash__(self):
        return hash((self.name, self.os, self.arch))

    @staticmethod
    def _parse_os(os):
        os = os.lower()

        if os == 'darwin':
            return 'macos'
        if os in ('win', 'win32', 'win64'):
            return 'windows'
        if os.startswith('cygwin'):
            return 'cygwin'

        return os


def which(prog):
    if os.path.exists(prog) and os.access(prog, os.X_OK):
        return prog

    # Ищем в $PATH только простые команды, без путей.
    if os.path.dirname(prog) != '':
        return None

    path = os.getenv('PATH', '')

    pathext = os.environ.get('PATHEXT')
    # На Windows %PATHEXT% указывает на список расширений, которые нужно проверять
    # при поиске команды в пути. Точное совпадение без расширения имеет приоритет.
    pathext = [''] if pathext is None else [''] + pathext.lower().split(os.pathsep)

    for dir_ in path.split(os.path.pathsep):
        for ext in pathext:
            p = os.path.join(dir_, prog + ext)
            if os.path.exists(p) and os.path.isfile(p) and os.access(p, os.X_OK):
                return p

    return None


def get_stdout(command):
    stdout, code = get_stdout_and_code(command)
    return stdout if code == 0 else None


def get_stdout_and_code(command):
    # noinspection PyBroadException
    try:
        process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, _ = process.communicate()
        return stdout, process.returncode
    except Exception:
        return None, None


def to_strings(o):
    if isinstance(o, (list, tuple)):
        for s in o:
            for ss in to_strings(s):
                yield ss
    else:
        if o is not None:
            if isinstance(o, bool):
                yield 'yes' if o else 'no'
            elif isinstance(o, (str, int)):
                yield str(o)
            else:
                raise ConfigureError('Unexpected value {} {}'.format(type(o), o))


def emit(key, *value):
    print '{0}={1}'.format(key, ' '.join(to_strings(value)))


def append(key, *value):
    print '{0}+={1}'.format(key, ' '.join(to_strings(value)))


def emit_big(text):
    prefix = None
    first = True
    for line in text.split('\n'):
        if prefix is None:
            if not line:
                continue

            prefix = 0
            while prefix < len(line) and line[prefix] == ' ':
                prefix += 1

        if first:  # Be pretty, prepend an empty line before the output
            print
            first = False

        print line[prefix:]


class Variables(dict):
    def emit(self):
        for k in sorted(self.keys()):
            emit(k, self[k])

    def update_from_presets(self):
        for k in self.iterkeys():
            v = preset(k)
            if v is not None:
                self[k] = v

    def reset_if_any(self, value_check=None, reset_value=None):
        if value_check is None:
            def value_check(v_):
                return v_ is None

        if any(map(value_check, self.itervalues())):
            for k in self.iterkeys():
                self[k] = reset_value


def format_env(env, list_separator=':'):
    def format_value(value):
        return value if isinstance(value, str) else ('\\' + list_separator).join(value)

    def format(kv):
        return '${env:"%s=%s"}' % (kv[0], format_value(kv[1]))

    return ' '.join(map(format, sorted(env.iteritems())))


# TODO(somov): Проверить, используется ли это. Может быть, выпилить.
def userify_presets(presets, keys):
    for key in keys:
        user_key = 'USER_{}'.format(key)
        values = [presets.pop(key, None), presets.get(user_key)]
        presets[user_key] = ' '.join(filter(None, values))


def preset(key, default=None):
    return opts().presets.get(key, default)


def is_positive(key):
    return is_positive_str(preset(key, ''))


def is_positive_str(s):
    return s.lower() in ('yes', 'true', 'on', '1')


def is_negative(key):
    return is_negative_str(preset(key, ''))


def is_negative_str(s):
    return s.lower() in ('no', 'false', 'off', '0')


def to_bool(s, default=None):
    if isinstance(s, basestring):
        if is_positive_str(s):
            return True
        if is_negative_str(s):
            return False
    if default is None:
        raise ConfigureError('{} is not a bool value'.format(s))
    return default


def select(selectors, default=None, no_default=False):
    for enabled, value in selectors:
        if enabled:
            return value
    if no_default:
        raise ConfigureError()
    return default


def select_multiple(selectors):
    for enabled, value in selectors:
        if enabled:
            yield value


def unique(it):
    known = set()
    for i in it:
        if i not in known:
            known.add(i)
            yield i


class Options(object):
    def __init__(self, argv):
        def parse_presets(raw_presets):
            presets = {}
            for p in raw_presets:
                toks = p.split('=', 1)
                name = toks[0]
                value = toks[1] if len(toks) >= 2 else ''
                presets[name] = value
            return presets

        parser = optparse.OptionParser(add_help_option=False)
        opt_group = optparse.OptionGroup(parser, 'Conf script options')
        opt_group.add_option('--toolchain-params', dest='toolchain_params', action='store', help='Set toolchain params via file')
        opt_group.add_option('-D', '--preset', dest='presets', action='append', default=[], help='set or override presets')
        opt_group.add_option('-l', '--local-distbuild', dest='local_distbuild', action='store_true', default=False, help='conf for local distbuild')
        parser.add_option_group(opt_group)

        self.options, self.arguments = parser.parse_args(argv)

        argv = self.arguments
        if len(argv) < 4:
            print >> sys.stderr, 'Usage: ArcRoot, --BuildType--, Verbosity, [Path to local.ymake]'
            sys.exit(1)

        self.arcadia_root = argv[1]
        init_logger(argv[3] == 'verbose')

        # Эти переменные не надо использоваться напрямую. Их значения уже разбираются в других местах.
        self.build_type = argv[2].lower()
        self.local_distbuild = self.options.local_distbuild
        self.toolchain_params = self.options.toolchain_params

        self.presets = parse_presets(self.options.presets)
        userify_presets(self.presets, ('CFLAGS', 'CXXFLAGS', 'CONLYFLAGS', 'LDFLAGS'))

    Instance = None


def opts():
    if Options.Instance is None:
        Options.Instance = Options(sys.argv)
    return Options.Instance


class Profiler(object):
    Generic = 'generic'
    GProf = 'gprof'


class Arcadia(object):
    def __init__(self, root):
        self.root = root


class Build(object):
    def __init__(self, arcadia, build_type, toolchain_params, force_ignore_local_files=False):
        self.arcadia = arcadia
        self.params = self._load_json_from_base64(toolchain_params)
        self.build_type = build_type

        platform = self.params['platform']
        self.host = Platform.from_json(platform['host'])
        self.target = Platform.from_json(platform['target'])

        self.tc = self._get_toolchain_options()

        # TODO(somov): Удалить, когда перестанет использоваться.
        self.build_system = 'ymake'

        self.ignore_local_files = False

        dist_prefix = 'dist-'
        if self.build_type.startswith(dist_prefix):
            self.build_system = 'distbuild'
            self.build_type = self.build_type[len(dist_prefix):]

        if force_ignore_local_files:
            self.ignore_local_files = True

        if self.is_ide_build_type(self.build_type):
            self.ignore_local_files = True

        self.pic = not is_positive('FORCE_NO_PIC')

    @property
    def host_target(self):
        return self.host, self.target

    def print_build(self):
        self._print_build_settings()

        host_os = System(self.host)
        host_os.print_host_settings()

        target_os = System(self.target)
        target_os.print_target_settings()

        if self.pic:
            emit('PIC', 'yes')

        emit('COMPILER_ID', self.tc.type.upper())

        if self.is_valgrind:
            emit('WITH_VALGRIND', 'yes')

        toolchain_type, compiler_type, linker_type = Compilers[self.tc.type]
        toolchain = toolchain_type(self.tc, self)
        compiler = compiler_type(self.tc, self)
        linker = linker_type(self.tc, self)

        toolchain.print_toolchain()
        compiler.print_compiler()
        linker.print_linker()

        self._print_other_settings(compiler)

    def _print_build_settings(self):
        emit('BUILD_TYPE', self.build_type.upper())
        emit('BT_' + self.build_type.upper().replace('-', '_'), 'yes')

        if self.build_system == 'distbuild':
            emit('DISTBUILD', 'yes')
        elif self.build_system != 'ymake':
            raise ConfigureError()

        python_bin = preset('BUILD_PYTHON_BIN', '$(PYTHON)/python')

        emit('YMAKE_PYTHON', python_bin)
        emit('YMAKE_UNPICKLER', python_bin, '$ARCADIA_ROOT/build/plugins/_unpickler.py')

    @property
    def is_release(self):
        # TODO(somov): Проверить, бывают ли тут суффиксы на самом деле
        return self.build_type in ('release', 'relwithdebinfo', 'minsizerel', 'profile', 'gprof') or self.build_type.endswith('-release')

    @property
    def is_debug(self):
        return self.build_type in ('debug', 'debugnoasserts') or self.build_type.endswith('-debug')

    @property
    def is_size_optimized(self):
        return self.build_type == 'minsizerel'

    @property
    def is_coverage(self):
        return self.build_type == 'coverage'

    @property
    def is_sanitized(self):
        sanitizer = preset('SANITIZER_TYPE')
        return bool(sanitizer) and not is_negative_str(sanitizer)

    @property
    def with_ndebug(self):
        return self.build_type in ('release', 'minsizerel', 'valgrind-release', 'profile', 'gprof', 'debugnoasserts')

    @property
    def is_valgrind(self):
        return self.build_type == 'valgrind' or self.build_type == 'valgrind-release'

    @property
    def is_ide(self):
        return self.is_ide_build_type(self.build_type)

    @property
    def profiler_type(self):
        if self.build_type == 'profile':
            return Profiler.Generic
        elif self.build_type == 'gprof':
            return Profiler.GProf
        else:
            return None

    @staticmethod
    def is_ide_build_type(build_type):
        return build_type == 'nobuild'

    def _get_toolchain_options(self):
        type_ = self.params['params']['type']

        if type_ == 'system_cxx':
            detector = CompilerDetector()
            detector.detect(self.params['params'].get('c_compiler'), self.params['params'].get('cxx_compiler'))
            type_ = detector.type
        else:
            detector = None

        if type_ == 'msvc':
            return MSVCToolchainOptions(self, detector)
        else:
            return GnuToolchainOptions(self, detector)

    def _print_other_settings(self, compiler):
        host = self.host

        emit('USE_LOCAL_TOOLS', 'no' if self.ignore_local_files else 'yes')

        ragel = Ragel()
        ragel.configure_toolchain(self, compiler)
        ragel.print_variables()

        perl = Perl()
        perl.configure_local()
        perl.print_variables('LOCAL_')

        yasm = Yasm(self.target)
        yasm.configure()
        yasm.print_variables()

        swiftc = SwiftCompiler(self)
        swiftc.configure()
        swiftc.print_compiler()

        if host.is_linux or host.is_freebsd or host.is_macos or host.is_cygwin:
            if is_negative('USE_ARCADIA_PYTHON'):
                python = Python(self.tc)
                python.configure_posix()
                python.print_variables()

        Cuda(self).print_()

        if self.ignore_local_files or host.is_windows or is_positive('NO_SVN_DEPENDS'):
            emit('SVN_DEPENDS')
            emit('SVN_DEPENDS_CACHE__NO_UID__')
        else:
            def find_svn():
                for i in range(0, 3):
                    for path in (['.svn', 'wc.db'], ['.svn', 'entries'], ['.git', 'logs', 'HEAD']):
                        path_parts = [os.pardir] * i + path
                        full_path = os.path.join(self.arcadia.root, *path_parts)
                        # HACK(somov): No "normpath" here. ymake fails with the "source file name is outside the build tree" error
                        # when .svn/wc.db found in "trunk" instead of "arcadia". But $ARCADIA_ROOT/../.svn/wc.db is ok.
                        if os.path.exists(full_path):
                            out_path = os.path.join('${ARCADIA_ROOT}', *path_parts)
                            return '${input;hide:"%s"}' % out_path

                # Special processing for arc repository since .arc may be a symlink.
                dot_arc = os.path.realpath(os.path.join(self.arcadia.root, '.arc'))
                full_path = os.path.join(dot_arc, 'TREE')
                if os.path.exists(full_path):
                    out_path = os.path.join('${ARCADIA_ROOT}', os.path.relpath(full_path, self.arcadia.root))
                    return '${input;hide:"%s"}' % out_path

                return ''

            emit('SVN_DEPENDS', find_svn())
            emit('SVN_DEPENDS_CACHE__NO_UID__', '${hide;kv:"disable_cache"}')

    @staticmethod
    def _load_json_from_base64(base64str):
        """
        :rtype: dict[str, Any]
        """

        def un_unicode(o):
            if isinstance(o, unicode):
                return o.encode('utf-8')
            if isinstance(o, list):
                return [un_unicode(oo) for oo in o]
            if isinstance(o, dict):
                return {un_unicode(k): un_unicode(v) for k, v in o.iteritems()}
            return o

        return un_unicode(json.loads(base64.b64decode(base64str)))


class YMake(object):
    def __init__(self, arcadia):
        self.arcadia = arcadia

    @staticmethod
    def print_presets():
        if opts().presets:
            print '# Variables set from command line by -D options'
            for key in opts().presets:
                print '{0}={1}'.format(key, opts().presets[key])

    def print_core_conf(self):
        with open(self._find_core_conf(), 'r') as fin:
            print fin.read()

    def print_settings(self):
        emit('ARCADIA_ROOT', self.arcadia.root)

    @staticmethod
    def _find_core_conf():
        script_dir = os.path.dirname(__file__)
        full_path = os.path.join(script_dir, 'ymake.core.conf')
        if os.path.exists(full_path):
            return full_path
        return None


class System(object):
    def __init__(self, platform):
        """
        :type platform: Platform
        """
        self.platform = platform

    def print_windows_target_const(self):
        # TODO(somov): Remove this variables, use generic OS/arch variables in makelists.
        emit('WINDOWS', 'yes')
        emit('WIN32', 'yes')
        if self.platform.is_64_bit == 64:
            emit('WIN64', 'yes')

    def print_nix_target_const(self):
        emit('JAVA_INCLUDE', '-I{0}/include -I{0}/include/{1}'.format('/usr/lib/jvm/default-java', self.platform.os_compat))

        emit('UNIX', 'yes')
        emit('REALPRJNAME')
        emit('SONAME')

    @staticmethod
    def print_nix_host_const():
        emit('WRITE_COMMAND', '/bin/echo', '-e')

        print '''
when ($USE_PYTHON) {
    C_DEFINES+= -DUSE_PYTHON
}'''

    @staticmethod
    def print_freebsd_const():
        emit('FREEBSD_VER', '9')
        emit('FREEBSD_VER_MINOR', '0')

        print '''
when (($USEMPROF == "yes") || ($USE_MPROF == "yes")) {
    C_LIBRARY_PATH+=-L/usr/local/lib
    C_SYSTEM_LIBRARIES_INTERCEPT+=-lc_mp
}
when (($USEMPROF == "yes") || ($USE_MPROF == "yes")) {
    C_DEFINES+= -DUSE_MPROF
}
'''

    @staticmethod
    def print_linux_const():
        print '''
when (($USEMPROF == "yes") || ($USE_MPROF == "yes")) {
    C_SYSTEM_LIBRARIES_INTERCEPT+=-ldmalloc
}
'''

    def print_target_settings(self):
        emit('TARGET_PLATFORM', self.platform.os_compat)
        emit('HARDWARE_ARCH', '32' if self.platform.is_32_bit else '64')
        emit('HARDWARE_TYPE', self.platform.arch)

        for variable in self.platform.arch_variables:
            emit(variable, 'yes')

        for variable in self.platform.os_variables:
            emit(variable, 'yes')

        if self.platform.is_posix:
            self.print_nix_target_const()
            if self.platform.is_linux:
                self.print_linux_const()
            elif self.platform.is_freebsd:
                self.print_freebsd_const()
        elif self.platform.is_windows:
            self.print_windows_target_const()

        self.print_target_shortcuts()

    # Misc target arch-related shortcuts
    def print_target_shortcuts(self):
        if preset('HAVE_MKL') is None:
            print 'HAVE_MKL=no'
            if self.platform.is_linux:
                print '''
  when ($ARCH_X86_64 && !$SANITIZER_TYPE) {
      HAVE_MKL=yes
  }
'''

    def print_host_settings(self):
        emit('HOST_PLATFORM', self.platform.os_compat)
        if not self.platform.is_windows:
            self.print_nix_host_const()

        for variable in itertools.chain(self.platform.os_variables, self.platform.arch_variables):
            emit('HOST_{var}'.format(var=variable), 'yes')


class CompilerDetector(object):
    def __init__(self):
        self.type = None
        self.c_compiler = None
        self.cxx_compiler = None
        self.version_list = None

    @staticmethod
    def preprocess_source(compiler, source):
        # noinspection PyBroadException
        try:
            fd, path = tempfile.mkstemp(suffix='.cpp')
            try:
                with os.fdopen(fd, 'wb') as output:
                    output.write(source)
                stdout, code = get_stdout_and_code([compiler, '-E', path])
            finally:
                os.remove(path)
            return stdout, code

        except Exception as e:
            logger.debug('Preprocessing failed: %s', e)
            return None, None

    @staticmethod
    def get_compiler_vars(compiler, names):
        prefix = '____YA_VAR_'
        source = '\n'.join(['{prefix}{name}={name}\n'.format(prefix=prefix, name=n) for n in names])

        # Некоторые препроцессоры возвращают ненулевой код возврата. Поэтому его проверять нельзя.
        # Мы можем только удостовериться после разбора stdout, что в нём
        # присутствовала хотя бы одна подставленная переменная.
        # TODO(somov): Исследовать, можно ли проверять ограниченный набор кодов возврата.
        stdout, _ = CompilerDetector.preprocess_source(compiler, source)

        if stdout is None:
            return None

        vars_ = {}
        for line in stdout.split('\n'):
            parts = line.split('=', 1)
            if len(parts) == 2 and parts[0].startswith(prefix):
                name, value = parts[0][len(prefix):], parts[1]
                if value == name:
                    continue  # Preprocessor variable was not substituted
                vars_[name] = value

        return vars_

    def detect(self, c_compiler=None, cxx_compiler=None):
        c_compiler = c_compiler or os.environ.get('CC')
        cxx_compiler = cxx_compiler or os.environ.get('CXX') or c_compiler
        c_compiler = c_compiler or cxx_compiler

        logger.debug('e=%s', os.environ)
        if c_compiler is None:
            raise ConfigureError('Custom compiler was requested but not specified')

        c_compiler_path = which(c_compiler)

        clang_vars = ['__clang_major__', '__clang_minor__', '__clang_patchlevel__']
        gcc_vars = ['__GNUC__', '__GNUC_MINOR__', '__GNUC_PATCHLEVEL__']
        msvc_vars = ['_MSC_VER']
        apple_var = '__apple_build_version__'

        compiler_vars = self.get_compiler_vars(c_compiler_path, clang_vars + [apple_var] + gcc_vars + msvc_vars)

        if not compiler_vars:
            raise ConfigureError('Could not determine custom compiler version: {}'.format(c_compiler))

        def version(version_names):
            def iter_version():
                for name in version_names:
                    yield int(compiler_vars[name])

            # noinspection PyBroadException
            try:
                return list(iter_version())
            except Exception:
                return None

        clang_version = version(clang_vars)
        apple_build = apple_var in compiler_vars
        # TODO(somov): Учитывать номера версий сборки Apple компилятора Clang.
        _ = apple_build
        gcc_version = version(gcc_vars)
        msvc_version = version(msvc_vars)

        if clang_version:
            logger.debug('Detected Clang version %s', clang_version)
            self.type = 'clang'
        elif gcc_version:
            logger.debug('Detected GCC version %s', gcc_version)
            # TODO(somov): Переименовать в gcc.
            self.type = 'gnu'
        elif msvc_version:
            logger.debug('Detected MSVC version %s', msvc_version)
            self.type = 'msvc'
        else:
            raise ConfigureError('Could not determine custom compiler type: {}'.format(c_compiler))

        self.version_list = clang_version or gcc_version or msvc_version

        self.c_compiler = c_compiler_path
        self.cxx_compiler = cxx_compiler and which(cxx_compiler) or c_compiler_path


class ToolchainOptions(object):
    def __init__(self, build, detector):
        """
        :type build: Build
        """
        self.host = build.host
        self.target = build.target

        tc_json = build.params

        logger.debug('Toolchain host %s', self.host)
        logger.debug('Toolchain target %s', self.target)
        logger.debug('Toolchain json %s', DebugString(lambda: json.dumps(tc_json, indent=4, sort_keys=True)))

        self.params = tc_json['params']
        self._name = tc_json.get('name', 'theyknow')

        if detector:
            self.type = detector.type
            self.from_arcadia = False

            self.c_compiler = detector.c_compiler
            self.cxx_compiler = detector.cxx_compiler
            self.compiler_version_list = detector.version_list
            self.compiler_version = '.'.join(map(str, self.compiler_version_list))

        else:
            self.type = self.params['type']
            self.from_arcadia = True

            self.c_compiler = self.params['c_compiler']
            self.cxx_compiler = self.params['cxx_compiler']

            # TODO(somov): Требовать номер версии всегда.
            self.compiler_version = self.params.get('gcc_version') or self.params.get('version') or '0'
            self.compiler_version_list = map(int, self.compiler_version.split('.'))

        # TODO(somov): Посмотреть, можно ли спрятать это поле.
        self.name_marker = '$(%s)' % self.params.get('match_root', self._name.upper())

        self.arch_opt = self.params.get('arch_opt', [])
        self.triplet_opt = self.params.get('triplet_opt', {})
        self.target_opt = self.params.get('target_opt', [])

        # TODO(somov): Убрать чтение настройки из os.environ.
        self.werror_mode = preset('WERROR_MODE') or os.environ.get('WERROR_MODE') or self.params.get('werror_mode') or 'compiler_specific'

        # default C++ standard is set here, some older toolchains might need to redefine it
        self.cxx_std = self.params.get('cxx_std', 'c++1z')

        self._env = tc_json.get('env', {})

        logger.debug('c_compiler=%s', self.c_compiler)
        logger.debug('cxx_compiler=%s', self.cxx_compiler)

        self.compiler_platform_projects = self.target.find_in_dict(self.params.get('platform'), [])

    def version_at_least(self, *args):
        return args <= tuple(self.compiler_version_list)

    @property
    def is_gcc(self):
        return self.type == 'gnu'

    @property
    def is_clang(self):
        return self.type == 'clang'

    @property
    def is_from_arcadia(self):
        return self.from_arcadia

    def get_env(self, convert_list=None):
        convert_list = convert_list or (lambda x: x)
        r = {}
        for k, v in self._env.iteritems():
            if isinstance(v, str):
                r[k] = v
            elif isinstance(v, list):
                r[k] = convert_list(v)
            else:
                logger.debug('Unexpected values in environment: %s', self._env)
                raise ConfigureError('Internal error')
        return r


class GnuToolchainOptions(ToolchainOptions):
    def __init__(self, build, detector):
        super(GnuToolchainOptions, self).__init__(build, detector)

        self.ar = self.params.get('ar')
        self.ar_plugin = self.params.get('ar_plugin')
        self.inplace_tools = self.params.get('inplace_tools', False)
        self.strip = self.params.get('strip')
        self.objcopy = self.params.get('objcopy')
        self.isystem = self.params.get('isystem')

        self.dwarf_tool = self.target.find_in_dict(self.params.get('dwarf_tool'))

        # TODO(somov): Унифицировать формат sys_lib
        self.sys_lib = self.params.get('sys_lib', {})
        if isinstance(self.sys_lib, dict):
            self.sys_lib = self.target.find_in_dict(self.sys_lib, [])

        self.os_sdk = preset('OS_SDK') or self._default_os_sdk()
        self.os_sdk_local = self.os_sdk == 'local'

    def _default_os_sdk(self):
        if self.target.is_linux:
            if self.target.is_armv8:
                return 'ubuntu-16'

            if self.target.is_armv7:
                return 'ubuntu-16'

            if self.target.is_ppc64le:
                return 'ubuntu-14'

            # Default OS SDK for Linux builds
            return 'ubuntu-12'


class Toolchain(object):
    def __init__(self, tc, build):
        """
        :type tc: ToolchainOptions
        :type build: Build
        """
        self.tc = tc
        self.build = build
        self.platform_projects = self.tc.compiler_platform_projects

    def print_toolchain(self):
        if self.platform_projects:
            emit('COMPILER_PLATFORM', list(unique(self.platform_projects)))


class Compiler(object):
    def __init__(self, tc, compiler_variable):
        self.compiler_variable = compiler_variable
        self.tc = tc

    def print_compiler(self):
        # CLANG and CLANG_VER variables
        emit(self.compiler_variable, 'yes')
        emit('{}_VER'.format(self.compiler_variable), self.tc.compiler_version)


class GnuToolchain(Toolchain):
    def __init__(self, tc, build):
        """
        :type tc: GnuToolchainOptions
        :type build: Build
        """

        def get_os_sdk(target):
            if target.is_macos:
                return '$MACOS_SDK_RESOURCE_GLOBAL/MacOSX10.11.sdk'
            elif target.is_yocto or target.is_yocto_lg_wk7y or target.is_yocto_jbl_portable_music or target.is_yocto_aacrh64_lightcomm_mt8516:
                return '$YOCTO_SDK_RESOURCE_GLOBAL'
            return '$OS_SDK_ROOT_RESOURCE_GLOBAL'

        super(GnuToolchain, self).__init__(tc, build)
        self.tc = tc

        host = build.host
        target = build.target

        self.c_flags_platform = list(tc.target_opt)

        self.default_os_sdk_root = get_os_sdk(target)

        self.env = self.tc.get_env()

        self.swift_flags_platform = []
        self.swift_lib_path = None

        if self.tc.is_from_arcadia:
            for lib_path in build.host.library_path_variables:
                self.env.setdefault(lib_path, []).append('{}/lib'.format(self.tc.name_marker))

        swift_target = select(default=None, selectors=[
            (target.is_ios and target.is_x86_64, 'x86_64-apple-ios9-simulator'),
            (target.is_ios and target.is_x86, 'i386-apple-ios9-simulator'),
            (target.is_ios and target.is_armv8, 'arm64-apple-ios9'),
            (target.is_ios and target.is_armv7, 'armv7-apple-ios9'),
        ])
        if swift_target:
            self.swift_flags_platform += ['-target', swift_target]

        if self.tc.is_from_arcadia:
            self.swift_lib_path = select(default=None, selectors=[
                (host.is_macos and target.is_ios and (target.is_x86_64 or target.is_x86), '$SWIFT_XCODE_TOOLCHAIN_ROOT_RESOURCE_GLOBAL/usr/lib/swift/iphonesimulator'),
                (host.is_macos and target.is_ios and (target.is_armv8 or target.is_armv7), '$SWIFT_XCODE_TOOLCHAIN_ROOT_RESOURCE_GLOBAL/usr/lib/swift/iphoneos'),
            ])

        if self.tc.is_clang:
            target_triple = self.tc.triplet_opt.get(target.arch, None)
            if not target_triple:
                target_triple = select(default=None, selectors=[
                    (target.is_linux and target.is_x86_64, 'x86_64-linux-gnu'),
                    (target.is_linux and target.is_armv8, 'aarch64-linux-gnu'),
                    (target.is_linux and target.is_armv7hf, 'arm-linux-gnueabihf'),
                    (target.is_linux and target.is_armv7, 'arm-linux-gnueabi'),
                    (target.is_linux and target.is_ppc64le, 'powerpc64le-linux-gnu'),
                    (target.is_apple and target.is_x86, 'i386-apple-darwin14'),
                    (target.is_apple and target.is_x86_64, 'x86_64-apple-darwin14'),
                    (target.is_apple and target.is_armv7, 'armv7-apple-darwin14'),
                    (target.is_apple and target.is_armv8, 'arm64-apple-darwin14'),
                    (target.is_yocto and target.is_armv7, 'arm-poky-linux-gnueabi'),
                    (target.is_android and target.is_x86, 'i686-linux-android16'),
                    (target.is_android and target.is_x86_64, 'x86_64-linux-android21'),
                    (target.is_android and target.is_armv7, 'armv7a-linux-androideabi16'),
                    (target.is_android and target.is_armv8, 'aarch64-linux-android21'),
                ])

            if target_triple:
                self.c_flags_platform.append('--target={}'.format(target_triple))

        if self.tc.isystem:
            for root in list(self.tc.isystem):
                self.c_flags_platform.extend(['-isystem', root])

        if target.is_android:
            self.c_flags_platform.extend(['-isystem', '{}/sources/cxx-stl/llvm-libc++abi/include'.format(self.tc.name_marker)])

        if target.is_cortex_a35:
            self.c_flags_platform.append('-mcpu=cortex-a35')

        elif target.is_cortex_a53:
            self.c_flags_platform.append('-mcpu=cortex-a53')

        elif target.is_armv7_neon:
            self.c_flags_platform.append('-mfpu=neon')

        if target.is_armv7 and build.is_size_optimized:
            # Enable ARM Thumb2 variable-length instruction encoding
            # to reduce code size
            self.c_flags_platform.append('-mthumb')

        if target.is_arm or target.is_ppc64le:
            # On linux, ARM and PPC default to unsigned char
            # However, Arcadia requires char to be signed
            self.c_flags_platform.append('-fsigned-char')

        if self.tc.is_clang or self.tc.is_gcc and self.tc.version_at_least(8, 2):
            target_flags = select(default=[], selectors=[
                (target.is_linux and target.is_ppc64le, ['-mcpu=power9', '-mtune=power9', '-maltivec']),
                (target.is_linux and target.is_armv8, ['-march=armv8a']),
                (target.is_macos, ['-mmacosx-version-min=10.11']),
                (target.is_ios and not target.is_intel, ['-mios-version-min=9.0']),
                (target.is_ios and target.is_intel, ['-mios-simulator-version-min=10.0']),
                (target.is_android and target.is_armv7, ['-march=armv7-a', '-mfloat-abi=softfp']),
                (target.is_android and target.is_armv8, ['-march=armv8-a']),
                (target.is_yocto and target.is_armv7, ['-march=armv7-a', '-mfpu=neon', '-mfloat-abi=hard', '-mcpu=cortex-a9', '-O1'])
            ])

            if target_flags:
                self.c_flags_platform.extend(target_flags)

            if target.is_ios:
                self.c_flags_platform.append('-D__IOS__=1')

            if self.tc.is_from_arcadia:
                if target.is_apple:
                    if target.is_ios:
                        self.setup_sdk(project='build/platform/ios_sdk', var='${IOS_SDK_ROOT_RESOURCE_GLOBAL}')
                    if target.is_macos:
                        self.setup_sdk(project='build/platform/macos_sdk', var='${MACOS_SDK_RESOURCE_GLOBAL}')

                    if not self.tc.inplace_tools:
                        self.setup_tools(project='build/platform/cctools', var='${CCTOOLS_ROOT_RESOURCE_GLOBAL}', bin='bin', ldlibs=None)

                if target.is_linux:
                    if not tc.os_sdk_local:
                        self.setup_sdk(project='build/platform/linux_sdk', var='$OS_SDK_ROOT_RESOURCE_GLOBAL')

                    if target.is_x86_64:
                        if host.is_linux and not self.tc.is_gcc:
                            self.setup_tools(project='build/platform/linux_sdk', var='$OS_SDK_ROOT_RESOURCE_GLOBAL', bin='usr/bin', ldlibs='usr/lib/x86_64-linux-gnu')
                        elif host.is_macos:
                            self.setup_tools(project='build/platform/binutils', var='$BINUTILS_ROOT_RESOURCE_GLOBAL', bin='x86_64-linux-gnu/bin', ldlibs=None)
                    elif target.is_ppc64le:
                        self.setup_tools(project='build/platform/linux_sdk', var='$OS_SDK_ROOT_RESOURCE_GLOBAL', bin='usr/bin', ldlibs='usr/x86_64-linux-gnu/powerpc64le-linux-gnu/lib')
                    elif target.is_armv8:
                        self.setup_tools(project='build/platform/linux_sdk', var='$OS_SDK_ROOT_RESOURCE_GLOBAL', bin='usr/bin', ldlibs='usr/lib/x86_64-linux-gnu')

                if target.is_yocto:
                    self.setup_sdk(project='build/platform/yocto_sdk/yocto_sdk', var='${YOCTO_SDK_ROOT_RESOURCE_GLOBAL}')
        elif self.tc.is_gcc and self.tc.version_at_least(6, 3):
            if self.tc.is_from_arcadia:
                if target.is_yocto_lg_wk7y:
                    self.c_flags_platform.extend(['-march=armv7ve', '-mfpu=neon-vfpv4', '-mfloat-abi=hard', '-mcpu=cortex-a7'])
                    self.setup_sdk(project='build/platform/yocto_sdk/yocto_armv7a_lg_wk7y_sdk', var='${YOCTO_SDK_ROOT_RESOURCE_GLOBAL}')
                elif target.is_yocto_jbl_portable_music:
                    self.c_flags_platform.extend(['-march=armv7ve', '-mfpu=neon-vfpv4', '-mfloat-abi=hard', '-mcpu=cortex-a7'])
                    self.setup_sdk(project='build/platform/yocto_sdk/yocto_armv7a_jbl_portable_music_sdk', var='${YOCTO_SDK_ROOT_RESOURCE_GLOBAL}')
                elif target.is_yocto_aacrh64_lightcomm_mt8516:
                    self.setup_sdk(project='build/platform/yocto_sdk/yocto_aarch64_lightcomm_mt8516', var='${YOCTO_SDK_ROOT_RESOURCE_GLOBAL}')

    def setup_sdk(self, project, var):
        self.platform_projects.append(project)
        self.c_flags_platform.append('--sysroot={}'.format(var))
        self.swift_flags_platform += ['-sdk', var]

    # noinspection PyShadowingBuiltins
    def setup_tools(self, project, var, bin, ldlibs):
        self.platform_projects.append(project)
        self.c_flags_platform.append('-B{}/{}'.format(var, bin))
        if ldlibs:
            for lib_path in self.build.host.library_path_variables:
                self.env.setdefault(lib_path, []).append('{}/{}'.format(var, ldlibs))

    def print_toolchain(self):
        super(GnuToolchain, self).print_toolchain()

        emit('TOOLCHAIN_ENV', format_env(self.env, list_separator=':'))
        emit('C_FLAGS_PLATFORM', self.c_flags_platform)
        emit('SWIFT_FLAGS_PLATFORM', self.swift_flags_platform)
        emit('SWIFT_LD_FLAGS', '-L{}'.format(self.swift_lib_path) if self.swift_lib_path else '')

        if preset('OS_SDK') is None:
            emit('OS_SDK', self.tc.os_sdk)
        emit('OS_SDK_ROOT', None if self.tc.os_sdk_local else self.default_os_sdk_root)


class GnuCompiler(Compiler):
    gcc_fstack = ['-fstack-protector']

    def __init__(self, tc, build):
        """
        :type tc: GnuToolchainOptions
        :type build: Build
        """
        compiler_variable = 'CLANG' if tc.is_clang else 'GCC'
        super(GnuCompiler, self).__init__(tc, compiler_variable)

        self.build = build
        self.host = self.build.host
        self.target = self.build.target
        self.tc = tc

        self.c_foptions = ['-fexceptions']
        self.c_warnings = ['-W', '-Wall', '-Wno-parentheses']
        self.cxx_warnings = [
            '-Woverloaded-virtual', '-Wno-invalid-offsetof', '-Wno-attributes',
        ]
        if not (self.tc.is_gcc and not self.tc.version_at_least(7)):
            self.cxx_warnings.extend([
                '-Wno-dynamic-exception-spec',  # IGNIETFERRO-282 some problems with lucid
                '-Wno-register',  # IGNIETFERRO-722 needed for contrib
            ])
        self.c_defines = [
            '-DFAKEID=$CPP_FAKEID', '-DARCADIA_ROOT=${ARCADIA_ROOT}', '-DARCADIA_BUILD_ROOT=${ARCADIA_BUILD_ROOT}',
            '-D_THREAD_SAFE', '-D_PTHREADS', '-D_REENTRANT', '-D_LIBCPP_ENABLE_CXX17_REMOVED_FEATURES',
            '-D_LARGEFILE_SOURCE', '-D__STDC_CONSTANT_MACROS', '-D__STDC_FORMAT_MACROS',
            '$CL_MACRO_INFO_DISABLE_CACHE__NO_UID__'
        ]
        self.c_flags = ['$CL_DEBUG_INFO_DISABLE_CACHE__NO_UID__']

        if not self.target.is_android:
            # There is no usable _FILE_OFFSET_BITS=64 support in Androids until API 21. And it's incomplete until at least API 24.
            # https://android.googlesource.com/platform/bionic/+/master/docs/32-bit-abi.md
            # Arcadia have API 16 for 32-bit Androids.
            self.c_defines.append('-D_FILE_OFFSET_BITS=64')

        if self.target.is_linux or self.target.is_cygwin or self.target.is_yocto_lg_wk7y or self.target.is_yocto_jbl_portable_music or self.target.is_yocto_aacrh64_lightcomm_mt8516:
            self.c_defines.append('-D_GNU_SOURCE')

        if self.target.is_ios:
            self.c_defines.extend(['-D_XOPEN_SOURCE', '-D_DARWIN_C_SOURCE'])
            if self.tc.version_at_least(7):
                self.c_foptions.append('$CLANG_ALIGNED_ALLOCATION_FLAG')
            else:
                self.c_warnings.append('-Wno-aligned-allocation-unavailable')
            if preset('MAPSMOBI_BUILD_TARGET') and self.target.is_arm:
                self.c_foptions.append('-fembed-bitcode')

        self.extra_compile_opts = []

        self.c_flags += self.tc.arch_opt + ['-pipe']

        self.sfdl_flags = ['-E', '-C', '-x', 'c++']

        if self.target.is_x86:
            self.c_flags.append('-m32')
        if self.target.is_x86_64:
            self.c_flags.append('-m64')

        self.debug_info_flags = ['-g']
        if self.target.is_linux:
            self.debug_info_flags.append('-ggnu-pubnames')

        self.cross_suffix = '' if is_positive('FORCE_NO_PIC') else '.pic'

        self.optimize = None

        self.configure_build_type()

        if self.tc.is_clang:
            self.sfdl_flags.append('-Qunused-arguments')

            self.cxx_warnings += [
                '-Wimport-preprocessor-directive-pedantic',
                '-Wno-c++17-extensions',
                '-Wno-exceptions',
                '-Wno-inconsistent-missing-override',
                '-Wno-undefined-var-template',
            ]

            if self.target.is_linux:
                self.c_foptions.append('-fuse-init-array')

            if self.tc.version_at_least(7):
                self.cxx_warnings.append('-Wno-return-std-move')

            if self.tc.version_at_least(8):
                self.cxx_warnings.extend((
                    '-Wno-address-of-packed-member',
                    '-Wno-defaulted-function-deleted',
                    '-Wno-enum-compare-switch',
                    '-Wno-pass-failed',
                ))
                if not self.target.is_ios:
                    self.c_foptions.append('$CLANG_ALIGNED_ALLOCATION_FLAG')

        if self.tc.is_gcc and self.tc.version_at_least(4, 9):
            self.c_foptions.append('-fno-delete-null-pointer-checks')
            self.c_foptions.append('-fabi-version=8')

    def configure_build_type(self):
        if self.build.is_valgrind:
            self.c_defines.append('-DWITH_VALGRIND=1')

        if self.build.is_debug:
            self.c_foptions.append('$FSTACK')

        if self.build.is_release:
            self.c_flags.append('$OPTIMIZE')
            if self.build.is_size_optimized:
                # -Oz is clang's more size-aggressive version of -Os
                # For ARM specifically, clang -Oz is on par with gcc -Os:
                # https://github.com/android/ndk/issues/133#issuecomment-365763507
                if self.tc.is_clang:
                    self.optimize = '-Oz'
                else:
                    self.optimize = '-Os'

                self.c_foptions.extend(['-ffunction-sections', '-fdata-sections'])
            else:
                self.optimize = '-O3'

        if self.build.with_ndebug:
            self.c_defines.append('-DNDEBUG')
        else:
            self.c_defines.append('-UNDEBUG')

        if self.build.is_coverage:
            self.c_foptions.extend(['-fprofile-arcs', '-ftest-coverage'])
        if self.build.profiler_type in (Profiler.Generic, Profiler.GProf):
            self.c_foptions.append('-fno-omit-frame-pointer')

        if self.build.profiler_type == Profiler.GProf:
            self.c_flags.append('-pg')

    def print_compiler(self):
        super(GnuCompiler, self).print_compiler()

        emit('C_COMPILER_UNQUOTED', self.tc.c_compiler)
        emit('C_COMPILER', '${quo:C_COMPILER_UNQUOTED}')
        emit('OPTIMIZE', self.optimize)
        emit('WERROR_MODE', self.tc.werror_mode)
        emit('FSTACK', self.gcc_fstack)
        append('C_DEFINES', self.c_defines)
        emit('DUMP_DEPS')
        emit('GCC_PREPROCESSOR_OPTS', '$DUMP_DEPS', '$C_DEFINES')
        append('C_WARNING_OPTS', self.c_warnings)
        append('CXX_WARNING_OPTS', self.cxx_warnings)

        # PIE is only valid for executables, while PIC implies a shared library
        # `-pie` with a shared library is either ignored or fails to link
        emit_big('''
            when ($PIC == "yes") {
                CFLAGS+=-fPIC
                LDFLAGS+=-fPIC
            }
            elsewhen ($PIE == "yes") {
                CFLAGS+=-fPIE
                LDFLAGS+=-fPIE -pie
            }''')

        append('CFLAGS', self.c_flags, '$DEBUG_INFO_FLAGS', self.c_foptions, '$C_WARNING_OPTS', '$GCC_PREPROCESSOR_OPTS', '$USER_CFLAGS', '$USER_CFLAGS_GLOBAL')
        append('CXXFLAGS', '$CFLAGS', '-std=' + self.tc.cxx_std, '$CXX_WARNING_OPTS', '$USER_CXXFLAGS')
        append('CONLYFLAGS', '$USER_CONLYFLAGS')
        emit('CXX_COMPILER_UNQUOTED', self.tc.cxx_compiler)
        emit('CXX_COMPILER', '${quo:CXX_COMPILER_UNQUOTED}')
        emit('NOGCCSTACKCHECK', 'yes')
        emit('USE_GCCFILTER', preset('USE_GCCFILTER') or 'yes')
        emit('USE_GCCFILTER_COLOR', preset('USE_GCCFILTER_COLOR') or 'yes')
        emit('SFDL_FLAG', self.sfdl_flags, '-o', '$SFDL_TMP_OUT')
        emit('WERROR_FLAG', '-Werror')
        # TODO(somov): Убрать чтение настройки из os.environ
        emit('USE_ARC_PROFILE', 'yes' if preset('USE_ARC_PROFILE') or os.environ.get('USE_ARC_PROFILE') else 'no')
        emit('DEBUG_INFO_FLAGS', self.debug_info_flags)

        emit_big('''
            when ($NO_WSHADOW == "yes") {
                C_WARNING_OPTS += -Wno-shadow
            }
            when ($NO_COMPILER_WARNINGS == "yes") {
                C_WARNING_OPTS = -w
                CXX_WARNING_OPTS = -Wno-register
            }
            when ($NO_OPTIMIZE == "yes") {
                OPTIMIZE = -O0
            }
            when ($SAVE_TEMPS ==  "yes") {
                CXXFLAGS += -save-temps
            }
            when ($NOGCCSTACKCHECK != "yes") {
                FSTACK += -fstack-check
            }
            ### @usage: MSVC_FLAGS([GLOBAL compiler_flag]* compiler_flags)
            ###
            ### Add the specified flags to the compilation line of C/C++files.
            ### Flags apply only if the compiler used is MSVC (cl.exe)
            macro MSVC_FLAGS(Flags...) {
                # TODO: FIXME
                ENABLE(UNUSED_MACRO)
            }''')

        c_builtins = [
            "-Wno-builtin-macro-redefined", '-D__DATE__=\\""Sep 31 2019\\""', '-D__TIME__=\\"00:00:00\\"',
            '-D__FILE__=\\""${qe;rootrel:SRC}\\""',
        ]
        c_debug_map = [
            # XXX does not support non-normalized paths
            "-fdebug-prefix-map=${ARCADIA_BUILD_ROOT}=/-B",
            "-fdebug-prefix-map=${ARCADIA_ROOT}=/-S",
            "-fdebug-prefix-map=$(TOOL_ROOT)=/-T",
        ]
        c_debug_map_cl = c_debug_map + [
            "-Xclang", "-fdebug-compilation-dir", "-Xclang", "/tmp",
        ]
        c_debug_map_light = [
            # XXX does not support non-normalized paths
            "-fdebug-prefix-map=${ARCADIA_BUILD_ROOT}=/-B",
        ]
        c_debug_map_light_cl = c_debug_map_light + [
            "-Xclang", "-fdebug-compilation-dir", "-Xclang", "/tmp",
        ]
        yasm_debug_map = [
            # XXX does not support non-normalized paths
            "--replace=${ARCADIA_BUILD_ROOT}=/-B",
            "--replace=${ARCADIA_ROOT}=/-S",
            "--replace=$(TOOL_ROOT)=/-T"
        ]
        emit_big('''
            when ($CONSISTENT_DEBUG == "yes") {{
                when ($CLANG == "yes") {{
                    CL_DEBUG_INFO_DISABLE_CACHE__NO_UID__={c_debug_cl}
                }}
                otherwise {{
                    CL_DEBUG_INFO_DISABLE_CACHE__NO_UID__={c_debug}
                }}
                YASM_DEBUG_INFO_DISABLE_CACHE__NO_UID__={yasm_debug}
            }}

            when ($CONSISTENT_DEBUG_LIGHT == "yes") {{
                when ($CLANG == "yes") {{
                    CL_DEBUG_INFO_DISABLE_CACHE__NO_UID__={c_debug_light_cl}
                }}
                otherwise {{
                    CL_DEBUG_INFO_DISABLE_CACHE__NO_UID__={c_debug_light}
                }}
                YASM_DEBUG_INFO_DISABLE_CACHE__NO_UID__={yasm_debug_light}
            }}

            when ($CONSISTENT_BUILD == "yes") {{
                CL_MACRO_INFO_DISABLE_CACHE__NO_UID__={macro}
            }}
        '''.format(c_debug=' '.join(c_debug_map),
                   c_debug_cl=' '.join(c_debug_map_cl),
                   yasm_debug=' '.join(yasm_debug_map),
                   c_debug_light=' '.join(c_debug_map_light),  # build_root substitution only
                   c_debug_light_cl=' '.join(c_debug_map_light_cl),  # build_root substitution only
                   yasm_debug_light=yasm_debug_map[0],  # build_root substitution only
                   macro=' '.join(c_builtins)))

        # TODO(somov): Check whether this specific architecture is needed.
        if self.target.arch == 'i386':
            append('CFLAGS', '-march=pentiumpro')
            append('CFLAGS', '-mtune=pentiumpro')

        append('BC_CFLAGS', '$CFLAGS')
        append('BC_CXXFLAGS', '$CXXFLAGS')

        append('C_DEFINES', '-D__LONG_LONG_SUPPORTED')

        emit('OBJECT_SUF', '$OBJ_SUF%s.o' % self.cross_suffix)
        emit('GCC_COMPILE_FLAGS', '$EXTRA_C_FLAGS -c -o $_COMPILE_OUTPUTS', '${input:SRC} ${pre=-I:_C__INCLUDE} ${pre=-I:INCLUDE}')
        emit('EXTRA_COVERAGE_OUTPUT', '${output;noauto;hide;suf=${OBJ_SUF}%s.gcno:SRC}' % self.cross_suffix)
        emit('YNDEXER_OUTPUT_FILE', '${output;noauto;suf=${OBJ_SUF}%s.ydx.pb2:SRC}' % self.cross_suffix)  # should be the last output

        if is_positive('DUMP_COMPILER_DEPS'):
            emit('DUMP_DEPS', '-MD', '${output;hide;noauto;suf=${OBJ_SUF}.o.d:SRC}')
        elif is_positive('DUMP_COMPILER_DEPS_FAST'):
            emit('DUMP_DEPS', '-E', '-M', '-MF', '${output;noauto;suf=${OBJ_SUF}.o.d:SRC}')

        if not self.build.is_coverage:
            emit('EXTRA_OUTPUT')
        else:
            emit('EXTRA_OUTPUT', '${output;noauto;hide;suf=${OBJ_SUF}%s.gcno:SRC}' % self.cross_suffix)

        append('EXTRA_OUTPUT')

        style = ['${hide;kv:"p CC"} ${hide;kv:"pc green"}']
        cxx_args = ['$GCCFILTER', '$YNDEXER_ARGS', '$CXX_COMPILER', '$C_FLAGS_PLATFORM', '$GCC_COMPILE_FLAGS', '$CXXFLAGS', '$EXTRA_OUTPUT', '$SRCFLAGS', '$TOOLCHAIN_ENV', '$YNDEXER_OUTPUT'] + style
        c_args = ['$GCCFILTER', '$YNDEXER_ARGS', '$C_COMPILER', '$C_FLAGS_PLATFORM', '$GCC_COMPILE_FLAGS', '$CFLAGS', '$CONLYFLAGS', '$EXTRA_OUTPUT', '$SRCFLAGS', '$TOOLCHAIN_ENV', '$YNDEXER_OUTPUT'] + style

        ignore_c_args_no_deps = ['$SRCFLAGS', '$YNDEXER_ARGS', '$YNDEXER_OUTPUT', '$EXTRA_OUTPUT', '$EXTRA_COVERAGE_OUTPUT']
        c_args_nodeps = [c if c != '$GCC_COMPILE_FLAGS' else '$EXTRA_C_FLAGS -c -o ${OUTFILE} ${SRC} ${pre=-I:INC}' for c in c_args if c not in ignore_c_args_no_deps]

        print 'macro _SRC_cpp(SRC, SRCFLAGS...) {\n .CMD=%s\n}' % ' '.join(cxx_args)
        print 'macro _SRC_c_nodeps(SRC, OUTFILE, INC...) {\n .CMD=%s\n}' % ' '.join(c_args_nodeps)
        print 'macro _SRC_c(SRC, SRCFLAGS...) {\n .CMD=%s\n}' % ' '.join(c_args)
        print 'macro _SRC_m(SRC, SRCFLAGS...) {\n .CMD=$SRC_c($SRC $SRCFLAGS)\n}'
        print 'macro _SRC_masm(SRC, SRCFLAGS...) {\n}'

        # fuzzing configuration
        if self.tc.is_clang and self.tc.version_at_least(5, 0):
            emit('LIBFUZZER_PATH',
                 'contrib/libs/libfuzzer8' if self.tc.version_at_least(8) else
                 'contrib/libs/libfuzzer7' if self.tc.version_at_least(7) else
                 'contrib/libs/libfuzzer6' if self.tc.version_at_least(6) else
                 'contrib/libs/libfuzzer-5.0')


class SwiftCompiler(object):
    def __init__(self, build):
        self.host = build.host
        self.compiler = None

    def configure(self):
        if self.host.is_macos:
            self.compiler = '$SWIFT_XCODE_TOOLCHAIN_ROOT_RESOURCE_GLOBAL/usr/bin/swiftc'

    def print_compiler(self):
        emit('SWIFT_COMPILER', self.compiler or '')


class Linker(object):
    BFD = 'bfd'
    LLD = 'lld'
    GOLD = 'gold'

    def __init__(self, tc, build):
        """
        :type tc: ToolchainOptions
        :type build: Build
        """
        self.tc = tc
        self.build = build

        if self.tc.is_from_arcadia and self.build.host.is_linux and not (self.build.target.is_apple or self.build.target.is_android or self.build.target.is_windows):

            if self.tc.is_clang:
                # DEVTOOLSSUPPORT-47 LLD cannot deal with extsearch/images/saas/base/imagesrtyserver
                self.type = Linker.GOLD if is_positive('USE_LTO') and not is_positive('MUSL') else Linker.LLD
            elif self.tc.is_gcc and (self.build.target.is_linux_armv7 or self.build.target.is_yocto_lg_wk7y or self.build.target.is_yocto_jbl_portable_music or self.build.target.is_yocto_aacrh64_lightcomm_mt8516):
                self.type = Linker.BFD
            else:
                self.type = Linker.GOLD
        else:
            self.type = None

    def print_linker(self):
        self._print_linker_selector()

    def _print_linker_selector(self):
        if self.type and self.tc.is_clang:
            # GCC does not support -fuse-ld.
            emit_big('''
                macro _USE_LINKER() {
                    DEFAULT(_LINKER_ID %(default_linker)s)

                    when ($NEED_PLATFORM_PEERDIRS == "yes") {
                        when ($_LINKER_ID == "bfd") {
                            PEERDIR+=build/platform/bfd
                        }
                        when ($_LINKER_ID == "gold") {
                            PEERDIR+=build/platform/gold
                        }
                        when ($_LINKER_ID == "lld") {
                            PEERDIR+=build/platform/lld
                        }
                    }
                }''' % {'default_linker': self.type})

        else:
            emit_big('''
                macro _USE_LINKER() {
                    SET(_LINKER_ID %(default_linker)s)
                }''' % {'default_linker': self.type})

        emit_big('''
            ### @usage: USE_LINKER_BFD()
            ### Use bfd linker for a program. This doesn't work in libraries
            macro USE_LINKER_BFD() {
                SET(_LINKER_ID bfd)
            }
            ### @usage: USE_LINKER_GOLD()
            ### Use gold linker for a program. This doesn't work in libraries
            macro USE_LINKER_GOLD() {
                SET(_LINKER_ID gold)
            }
            ### @usage: USE_LINKER_LLD()
            ### Use lld linker for a program. This doesn't work in libraries
            macro USE_LINKER_LLD() {
                SET(_LINKER_ID lld)
            }''')


class LD(Linker):
    def __init__(self, tc, build):
        """
        :type tc: GnuToolchainOptions
        :type build: Build
        """
        super(LD, self).__init__(tc, build)

        self.build = build
        self.host = self.build.host
        self.target = self.build.target
        self.tc = tc

        target = self.target

        self.ar = preset('AR') or self.tc.ar
        self.ar_plugin = self.tc.ar_plugin
        self.strip = self.tc.strip
        self.objcopy = self.tc.objcopy

        self.musl = Setting('MUSL', convert=to_bool)

        if self.ar is None:
            if target.is_apple:
                # Use libtool. cctools ar does not understand -M needed for archive merging
                self.ar = '${CCTOOLS_ROOT_RESOURCE_GLOBAL}/bin/libtool'
            elif self.tc.is_from_arcadia:
                if self.tc.is_clang:
                    self.ar = '{}/bin/llvm-ar'.format(self.tc.name_marker)
                if self.tc.is_gcc:
                    self.ar = '{}/gcc/bin/gcc-ar'.format(self.tc.name_marker)
            else:
                self.ar = 'ar'

        self.ld_flags = []

        if self.build.is_size_optimized:
            self.ld_flags.append('-Wl,--gc-sections')

        if self.musl.value:
            self.ld_flags.extend(['-Wl,--no-as-needed'])
        elif target.is_linux:
            self.ld_flags.extend(['-ldl', '-lrt', '-Wl,--no-as-needed'])
        elif target.is_android:
            self.ld_flags.extend(['-ldl', '-Wl,--no-as-needed'])
        elif target.is_macos:
            self.ld_flags.append('-Wl,-no_deduplicate')
            if not self.tc.is_clang:
                self.ld_flags.append('-Wl,-no_compact_unwind')

        self.thread_library = select([
            (target.is_linux or target.is_macos or target.is_yocto_lg_wk7y or self.target.is_yocto_jbl_portable_music or target.is_yocto_aacrh64_lightcomm_mt8516, '-lpthread'),
            (target.is_freebsd, '-lthr')
        ])

        self.rdynamic = None
        self.start_group = None
        self.end_group = None
        self.whole_archive = None
        self.no_whole_archive = None
        self.ld_stripflag = None
        self.use_stdlib = None
        self.soname_option = None
        self.dwarf_command = None
        self.libresolv = '-lresolv' if target.is_linux or target.is_macos or target.is_android else None

        if target.is_linux or target.is_android or target.is_freebsd:
            self.rdynamic = '-rdynamic'
            self.use_stdlib = '-nodefaultlibs'

        if target.is_linux or target.is_android or target.is_freebsd or target.is_cygwin or target.is_yocto_lg_wk7y or target.is_yocto_jbl_portable_music or target.is_yocto_aacrh64_lightcomm_mt8516:
            self.start_group = '-Wl,--start-group'
            self.end_group = '-Wl,--end-group'
            self.whole_archive = '-Wl,--whole-archive'
            self.no_whole_archive = '-Wl,--no-whole-archive'
            self.ld_stripflag = '-s'
            self.soname_option = '-soname'

        if target.is_macos or target.is_ios:
            self.use_stdlib = '-nodefaultlibs'
            self.soname_option = '-install_name'
            if not preset('NO_DEBUGINFO'):
                self.dwarf_command = '$DWARF_TOOL $TARGET -o ${output;pre=$MODULE_PREFIX$REALPRJNAME.dSYM/Contents/Resources/DWARF/$MODULE_PREFIX:REALPRJNAME}'

        if self.build.profiler_type == Profiler.GProf:
            self.ld_flags.append('-pg')

        if self.build.is_coverage:
            self.ld_flags.extend(('-fprofile-arcs', '-ftest-coverage'))

        # TODO(somov): Единое условие на coverage.
        if self.build.is_coverage or is_positive('GCOV_COVERAGE') or is_positive('CLANG_COVERAGE') or self.build.is_sanitized:
            self.use_stdlib = None

        self.ld_sdk = select(default=None, selectors=[
            (target.is_macos, '-Wl,-sdk_version,10.15'),
            (target.is_ios, '-Wl,-sdk_version,13.1'),
        ])

        if self.ld_sdk:
            self.ld_flags.append(self.ld_sdk)

        self.sys_lib = self.tc.sys_lib

        if target.is_android:
            if target.is_armv7 and self.type != Linker.LLD:
                self.sys_lib.append('-Wl,--fix-cortex-a8')
            if target.is_armv7:
                self.sys_lib.append('-lunwind')
            self.sys_lib.append('-lgcc')

        if self.tc.is_clang and not self.tc.version_at_least(4, 0) and target.is_linux_x86_64:
            self.sys_lib.append('-L/usr/lib/x86_64-linux-gnu')

    def print_linker(self):
        super(LD, self).print_linker()

        emit('AR_TOOL', self.ar)
        emit('AR_TYPE', 'AR' if 'libtool' not in self.ar else 'LIBTOOL')

        emit('STRIP_TOOL_VENDOR', self.strip)
        emit('OBJCOPY_TOOL_VENDOR', self.objcopy)

        append('LDFLAGS', '$USER_LDFLAGS', self.ld_flags)
        append('LDFLAGS_GLOBAL', '')

        emit('LD_STRIP_FLAG', self.ld_stripflag)
        emit('STRIP_FLAG')

        emit('C_LIBRARY_PATH')
        emit('C_SYSTEM_LIBRARIES_INTERCEPT')
        if self.musl.value:
            emit('C_SYSTEM_LIBRARIES', '-nostdlib')
        else:
            emit('C_SYSTEM_LIBRARIES', self.use_stdlib, self.thread_library, self.sys_lib, '-lc')

        emit('START_WHOLE_ARCHIVE_VALUE', self.whole_archive)
        emit('END_WHOLE_ARCHIVE_VALUE', self.no_whole_archive)

        if self.ld_sdk:
            emit('LD_SDK_VERSION', self.ld_sdk)

        dwarf_tool = self.tc.dwarf_tool
        if dwarf_tool is None and self.tc.is_clang and (self.target.is_macos or self.target.is_ios):
            dsymutil = '{}/bin/{}dsymutil'.format(self.tc.name_marker, '' if self.tc.version_at_least(7) else 'llvm-')
            dwarf_tool = '${YMAKE_PYTHON} ${input:"build/scripts/run_llvm_dsymutil.py"} ' + dsymutil
            if self.tc.version_at_least(5, 0):
                dwarf_tool += ' -flat'

        if dwarf_tool is not None:
            emit('DWARF_TOOL', dwarf_tool)

        emit('OBJADDE')

        emit_big('''
            EXPORTS_VALUE=
            when ($EXPORTS_FILE) {
                EXPORTS_VALUE=-Wl,--version-script=${input:EXPORTS_FILE}
            }''')

        exe_flags = [
            '$C_FLAGS_PLATFORM', '$BEFORE_PEERS', self.start_group, '${rootrel:PEERS}', self.end_group, '$AFTER_PEERS',
            '$EXPORTS_VALUE $LDFLAGS $LDFLAGS_GLOBAL $OBJADDE $OBJADDE_LIB',
            '$C_LIBRARY_PATH $C_SYSTEM_LIBRARIES_INTERCEPT $C_SYSTEM_LIBRARIES $STRIP_FLAG']

        arch_flag = '--arch={arch}'.format(arch=self.target.os_compat)
        soname_flag = '-Wl,{option},$SONAME'.format(option=self.soname_option)
        shared_flag = '-shared'
        exec_shared_flag = '-pie -fPIE -Wl,--unresolved-symbols=ignore-all -rdynamic' if self.target.is_linux else ''

        ld_env_style = '${cwd:ARCADIA_BUILD_ROOT} $TOOLCHAIN_ENV ${kv;hide:"p LD"} ${kv;hide:"pc light-blue"} ${kv;hide:"show_out"}'

        emit("GENERATE_MF",
             '$YMAKE_PYTHON', '${input:"build/scripts/generate_mf.py"}',
             '--build-root $ARCADIA_BUILD_ROOT --module-name $REALPRJNAME -o ${output;rootrel;pre=$MODULE_PREFIX;suf=$MODULE_SUFFIX.mf:REALPRJNAME}',
             '-t $MODULE_TYPE $NO_GPL_FLAG -Ya,lics $LICENSE_NAMES -Ya,peers ${rootrel:PEERS}'
             )

        # Program

        emit('LINK_SCRIPT_EXE_FLAGS')
        emit('REAL_LINK_EXE',
             '$YMAKE_PYTHON ${input:"build/scripts/link_exe.py"}', '$LINK_SCRIPT_EXE_FLAGS',
             '$GCCFILTER',
             '$CXX_COMPILER ${rootrel:SRCS_GLOBAL} $VCS_C_OBJ $AUTO_INPUT -o $TARGET', self.rdynamic, exe_flags,
             ld_env_style)

        # Executable Shared Library

        emit('REAL_LINK_EXEC_DYN_LIB_CMDLINE',
             '$YMAKE_PYTHON ${input:"build/scripts/link_dyn_lib.py"} --target $TARGET ${pre=--whole-archive :WHOLE_ARCHIVE_PEERS}', arch_flag, '$LINK_DYN_LIB_FLAGS',
             '$CXX_COMPILER ${rootrel:SRCS_GLOBAL} $VCS_C_OBJ $AUTO_INPUT -o $TARGET', exec_shared_flag, soname_flag, exe_flags,
             ld_env_style)
        emit_big('''
        macro REAL_LINK_EXEC_DYN_LIB_IMPL(WHOLE_ARCHIVE_PEERS...) {
            .CMD=$REAL_LINK_EXEC_DYN_LIB_CMDLINE
        }
        ''')
        emit('REAL_LINK_EXEC_DYN_LIB', '$REAL_LINK_EXEC_DYN_LIB_IMPL($_WHOLE_ARCHIVE_PEERS_VALUE)')

        # Shared Library

        emit('LINK_DYN_LIB_FLAGS')
        emit('REAL_LINK_DYN_LIB_CMDLINE',
             '$YMAKE_PYTHON ${input:"build/scripts/link_dyn_lib.py"} --target $TARGET ${pre=--whole-archive :WHOLE_ARCHIVE_PEERS}', arch_flag, '$LINK_DYN_LIB_FLAGS',
             '$CXX_COMPILER ${rootrel:SRCS_GLOBAL} $VCS_C_OBJ $AUTO_INPUT -o $TARGET', shared_flag, soname_flag, exe_flags,
             ld_env_style)
        emit_big('''
        macro REAL_LINK_DYN_LIB_IMPL(WHOLE_ARCHIVE_PEERS...) {
            .CMD=$REAL_LINK_DYN_LIB_CMDLINE
        }
        ''')
        emit('REAL_LINK_DYN_LIB', '$REAL_LINK_DYN_LIB_IMPL($_WHOLE_ARCHIVE_PEERS_VALUE)')

        if self.dwarf_command is None or self.target.is_ios:
            emit('DWARF_COMMAND')
        else:
            emit('DWARF_COMMAND', self.dwarf_command, ld_env_style)
        emit('LINK_EXE', '$GENERATE_MF && $GENERATE_VCS_C_INFO_NODEP && $REAL_LINK_EXE && $DWARF_COMMAND && $PACK_IOS_CMD')
        emit('LINK_DYN_LIB', '$GENERATE_MF && $GENERATE_VCS_C_INFO_NODEP && $REAL_LINK_DYN_LIB && $DWARF_COMMAND')
        emit('LINK_EXEC_DYN_LIB', '$GENERATE_MF && $GENERATE_VCS_C_INFO_NODEP && $REAL_LINK_EXEC_DYN_LIB && $DWARF_COMMAND')
        emit('SWIG_DLL_JAR_CMD', '$GENERATE_MF && $GENERATE_VCS_C_INFO_NODEP && $REAL_SWIG_DLL_JAR_CMD && $DWARF_COMMAND')

        archiver = '$YMAKE_PYTHON ${input:"build/scripts/link_lib.py"} ${quo:AR_TOOL} $AR_TYPE $ARCADIA_BUILD_ROOT %s' % (self.ar_plugin or 'None')

        # Static Library

        emit('LINK_LIB', '$GENERATE_MF &&', archiver, '$TARGET $AUTO_INPUT ${kv;hide:"p AR"}',
             '$TOOLCHAIN_ENV ${kv;hide:"pc light-red"} ${kv;hide:"show_out"}')

        # "Fat Object" : pre-linked global objects and static library with all dependencies
        def emit_link_fat_obj(cmd_name, *extended_flags):
            prefix = ['$GENERATE_MF && $GENERATE_VCS_C_INFO_NODEP &&',
                      '$YMAKE_PYTHON ${input:"build/scripts/link_fat_obj.py"} --build-root $ARCADIA_BUILD_ROOT']
            suffix = [arch_flag,
                      '-Ya,input $AUTO_INPUT $VCS_C_OBJ -Ya,global_srcs ${rootrel:SRCS_GLOBAL} -Ya,peers $PEERS',
                      '-Ya,linker $CXX_COMPILER $C_FLAGS_PLATFORM', self.ld_sdk, '-Ya,archiver', archiver,
                      '$TOOLCHAIN_ENV ${kv;hide:"p LD"} ${kv;hide:"pc light-blue"} ${kv;hide:"show_out"}']
            emit(cmd_name, *(prefix + list(extended_flags) + suffix))

        # TODO(somov): Проверить, не нужны ли здесь все остальные флаги компоновки (LDFLAGS и т. д.).
        emit_link_fat_obj('LINK_FAT_OBJECT', '--obj=$TARGET', '--lib=${output:REALPRJNAME.a}')
        emit_link_fat_obj('LINK_RECURSIVE_LIBRARY', '--lib=$TARGET', '--with-own-obj', '--with-global-srcs')

        emit('LIBRT', '-lrt')
        emit('MD5LIB', '-lcrypt')
        emit('LIBRESOLV', self.libresolv)
        emit('PROFFLAG', '-pg')


class MSVCToolchainOptions(ToolchainOptions):
    def __init__(self, build, detector):
        super(MSVCToolchainOptions, self).__init__(build, detector)

        # C:\Program Files (x86)\Microsoft Visual Studio\2017\Community\VC\Tools\MSVC\14.14.26428
        self.vc_root = None

        # C:\Program Files (x86)\Windows Kits\10\Include\10.0.14393.0
        self.kit_includes = None

        # C:\Program Files (x86)\Windows Kits\10\Lib\10.0.14393.0
        self.kit_libs = None

        self.under_wine = 'wine' in self.params
        self.system_msvc = 'system_msvc' in self.params
        self.ide_msvs = 'ide_msvs' in self.params
        self.use_clang = self.params.get('use_clang', False)
        self.use_arcadia_toolchain = self.params.get('use_arcadia_toolchain', False)

        self.sdk_version = None

        if self.ide_msvs:
            bindir = '$(VC_ExecutablePath_x64_x64)\\'
            self.c_compiler = bindir + 'cl.exe'
            self.cxx_compiler = self.c_compiler

            self.link = bindir + 'link.exe'
            self.lib = bindir + 'lib.exe'
            self.masm_compiler = bindir + 'ml64.exe'

            self.vc_root = None

            sdk_dir = '$(WindowsSdkDir)'
            self.sdk_version = '$(WindowsTargetPlatformVersion)'
            self.kit_includes = os.path.join(sdk_dir, 'Include', self.sdk_version)
            self.kit_libs = os.path.join(sdk_dir, 'Lib', self.sdk_version)

        elif detector:
            self.masm_compiler = which('ml64.exe')
            self.link = which('link.exe')
            self.lib = which('lib.exe')

            sdk_dir = os.environ.get('WindowsSdkDir')
            self.sdk_version = os.environ.get('WindowsSDKVersion').replace('\\', '')
            vc_install_dir = os.environ.get('VCToolsInstallDir')

            if any([x is None for x in (sdk_dir, self.sdk_version, vc_install_dir)]):
                raise ConfigureError('No %WindowsSdkDir%, %WindowsSDKVersion% or %VCINSTALLDIR% present. Please, run vcvars64.bat to setup preferred environment.')

            self.vc_root = os.path.normpath(vc_install_dir)
            self.kit_includes = os.path.normpath(os.path.join(sdk_dir, 'Include', self.sdk_version))
            self.kit_libs = os.path.normpath(os.path.join(sdk_dir, 'Lib', self.sdk_version))

            # TODO(somov): Определять автоматически self.version в этом случае

        else:
            self.sdk_version = '10.0.16299.0'
            sdk_dir = '$(WINDOWS_KITS-sbr:1379398385)'

            self.vc_root = self.name_marker if not self.use_clang else '$MSVC_FOR_CLANG_RESOURCE_GLOBAL'
            self.kit_includes = os.path.join(sdk_dir, 'Include', self.sdk_version)
            self.kit_libs = os.path.join(sdk_dir, 'Lib', self.sdk_version)

            bindir = os.path.join(self.vc_root, 'bin', 'Hostx64')

            tools_name = select(selectors=[
                (build.target.is_x86, 'x86'),
                (build.target.is_x86_64, 'x64'),
                (build.target.is_armv7, 'arm'),
            ])

            asm_name = select(selectors=[
                (build.target.is_x86, 'ml.exe'),
                (build.target.is_x86_64, 'ml64.exe'),
                (build.target.is_armv7, 'armasm.exe'),
            ])

            def prefix(_type, _path):
                if not self.under_wine:
                    return _path
                return '{wine} {type} $WINE_ENV ${{ARCADIA_ROOT}} ${{ARCADIA_BUILD_ROOT}} {path}'.format(wine='${YMAKE_PYTHON} ${input:\"build/scripts/run_msvc_wine.py\"} $(WINE_TOOL-sbr:1093314933)/bin/wine64 -v140', type=_type, path=_path)

            self.masm_compiler = prefix('masm', os.path.join(bindir, tools_name, asm_name))
            self.link = prefix('link', os.path.join(bindir, tools_name, 'link.exe'))
            self.lib = prefix('lib', os.path.join(bindir, tools_name, 'lib.exe'))


class MSVC(object):
    # noinspection PyPep8Naming
    class WIN32_WINNT(object):
        Macro = '_WIN32_WINNT'
        Windows7 = '0x0601'
        Windows8 = '0x0602'

    def __init__(self, tc, build):
        """
        :type tc: MSVCToolchainOptions
        :type build: Build
        """
        if not isinstance(tc, MSVCToolchainOptions):
            raise TypeError('Got {} ({}) instead of an MSVCToolchainOptions'.format(tc, type(tc)))

        self.build = build
        self.tc = tc


class MSVCToolchain(MSVC, Toolchain):
    def __init__(self, tc, build):
        """
        :type tc: MSVCToolchainOptions
        :param build: Build
        """
        Toolchain.__init__(self, tc, build)
        MSVC.__init__(self, tc, build)

        if self.tc.from_arcadia and not self.tc.ide_msvs:
            self.platform_projects.append('build/platform/msvc')
            if tc.under_wine:
                self.platform_projects.append('build/platform/wine')

    def print_toolchain(self):
        super(MSVCToolchain, self).print_toolchain()

        emit('TOOLCHAIN_ENV', format_env(self.tc.get_env(), list_separator=';'))

        if self.tc.sdk_version:
            emit('WINDOWS_KITS_VERSION', self.tc.sdk_version)

        # TODO(somov): Заглушка для тех мест, где C_FLAGS_PLATFORM используется
        # для любых платформ. Нужно унифицировать с GnuToolchain.
        emit('C_FLAGS_PLATFORM')

        if self.tc.under_wine:
            emit('WINE_ENV', format_env({'WINEPREFIX_SUFFIX': '4.0'}))


class MSVCCompiler(MSVC, Compiler):
    def __init__(self, tc, build):
        Compiler.__init__(self, tc, 'MSVC')
        MSVC.__init__(self, tc, build)

    def print_compiler(self):
        super(MSVCCompiler, self).print_compiler()

        target = self.build.target

        win32_winnt = self.WIN32_WINNT.Windows7

        warns_enabled = [
            4018,  # 'expression' : signed/unsigned mismatch
            4265,  # 'class' : class has virtual functions, but destructor is not virtual
            4296,  # 'operator' : expression is always false
            4431,  # missing type specifier - int assumed
        ]
        warns_as_error = [
            4013,  # 'function' undefined; assuming extern returning int
        ]
        warns_disabled = [
            4127,  # conditional expression is constant
            4200,  # nonstandard extension used : zero-sized array in struct/union
            4201,  # nonstandard extension used : nameless struct/union
            4351,  # elements of array will be default initialized
            4355,  # 'this' : used in base member initializer list
            4503,  # decorated name length exceeded, name was truncated
            4510,  # default constructor could not be generated
            4511,  # copy constructor could not be generated
            4512,  # assignment operator could not be generated
            4554,  # check operator precedence for possible error; use parentheses to clarify precedence
            4610,  # 'object' can never be instantiated - user defined constructor required
            4706,  # assignment within conditional expression
            4800,  # forcing value to bool 'true' or 'false' (performance warning)
            4996,  # The POSIX name for this item is deprecated
            4714,  # function marked as __forceinline not inlined
            4197,  # 'TAtomic' : top-level volatile in cast is ignored
            4245,  # 'initializing' : conversion from 'int' to 'ui32', signed/unsigned mismatch
            4324,  # 'ystd::function<void (uint8_t *)>': structure was padded due to alignment specifier
            5033,  # 'register' is no longer a supported storage class
        ]

        defines = [
            '/DFAKEID=$CPP_FAKEID',
            '/DWIN32',
            '/D_WIN32',
            '/D_WINDOWS',
            '/D_CRT_SECURE_NO_WARNINGS',
            '/D_CRT_NONSTDC_NO_WARNINGS',
            '/D_USE_MATH_DEFINES',
            '/D__STDC_CONSTANT_MACROS',
            '/D__STDC_FORMAT_MACROS',
            '/D_USING_V110_SDK71_',
            '/D_LIBCPP_ENABLE_CXX17_REMOVED_FEATURES',
        ]

        if target.is_x86_64:
            defines.extend(('/D_WIN64', '/DWIN64'))

        if target.is_armv7:
            defines.extend(('/D_ARM_WINAPI_PARTITION_DESKTOP_SDK_AVAILABLE', '/D__arm__'))

        winapi_unicode = False

        emit_big('''
            MSVC_INLINE_OPTIMIZED=yes
            when ($MSVC_INLINE_OPTIMIZED == "yes") {
                MSVC_INLINE_FLAG=/Zc:inline
            }
            when ($MSVC_INLINE_OPTIMIZED == "no") {
                MSVC_INLINE_FLAG=/Zc:inline-
            }''')

        flags = ['/nologo', '/Zm500', '/GR', '/bigobj', '/FC', '/EHs', '/errorReport:prompt', '$MSVC_INLINE_FLAG', '/utf-8']
        flags += self.tc.arch_opt

        c_warnings = ['/we{}'.format(code) for code in warns_as_error]
        c_warnings += ['/w1{}'.format(code) for code in warns_enabled]
        c_warnings += ['/wd{}'.format(code) for code in warns_disabled]
        cxx_warnings = []

        flags_debug = ['/Ob0', '/Od', '/D_DEBUG']
        flags_release = ['/Ox', '/Ob2', '/Oi', '/DNDEBUG']

        flags_c_only = []

        if self.tc.use_clang:
            cxx_warnings += [
                '-Woverloaded-virtual', '-Wno-invalid-offsetof', '-Wno-attributes',
                '-Wno-dynamic-exception-spec',  # IGNIETFERRO-282 some problems with lucid
                '-Wno-register',  # IGNIETFERRO-722 needed for contrib
                '-Wimport-preprocessor-directive-pedantic',
                '-Wno-c++17-extensions',
                '-Wno-exceptions',
                '-Wno-inconsistent-missing-override',
                '-Wno-undefined-var-template',
            ]
            if self.tc.ide_msvs:
                cxx_warnings += [
                    '-Wno-unused-command-line-argument',
                ]

        if target.is_armv7:
            masm_io = '-o ${output;suf=${OBJECT_SUF}:SRC} ${input;msvs_source:SRC}'
        else:
            masm_io = '/nologo /c /Fo${output;suf=${OBJECT_SUF}:SRC} ${input;msvs_source:SRC}'

        emit('OBJECT_SUF', '$OBJ_SUF.obj')
        emit('WIN32_WINNT', '{value}'.format(value=win32_winnt))
        defines.append('/D{name}=$WIN32_WINNT'.format(name=self.WIN32_WINNT.Macro))

        if winapi_unicode:
            defines += ['/DUNICODE', '/D_UNICODE']
        else:
            defines += ['/D_MBCS']

        # https://msdn.microsoft.com/en-us/library/abx4dbyh.aspx
        if is_positive('DLL_RUNTIME'):  # XXX
            flags_debug += ['/MDd']
            flags_release += ['/MD']
        else:
            flags_debug += ['/MTd']
            flags_release += ['/MT']

        vc_include = os.path.join(self.tc.vc_root, 'include') if not self.tc.ide_msvs else "$(VC_VC_IncludePath.Split(';')[0].Replace('\\','/'))"

        if not self.tc.ide_msvs:
            def include_flag(path):
                return '{flag}"{path}"'.format(path=path, flag='/I ' if not self.tc.use_clang else '-imsvc')

            for name in ('shared', 'ucrt', 'um', 'winrt'):
                flags.append(include_flag(os.path.join(self.tc.kit_includes, name)))
            flags.append(include_flag(vc_include))

        flags_msvs_only = []

        if self.tc.ide_msvs:
            if not self.tc.use_clang:
                flags_msvs_only += ['/FD', '/MP']
            debug_info_flags = '/Zi /FS'
        else:
            debug_info_flags = '/Z7'

        if self.tc.use_clang:
            emit('CLANG_CL', 'yes')
        if self.tc.ide_msvs:
            emit('IDE_MSVS', 'yes')
        if self.tc.use_arcadia_toolchain:
            emit('USE_ARCADIA_TOOLCHAIN', 'yes')

        emit('CXX_COMPILER', self.tc.cxx_compiler)
        emit('C_COMPILER', self.tc.c_compiler)
        emit('MASM_COMPILER', self.tc.masm_compiler)
        append('C_DEFINES', defines)
        emit('CFLAGS_DEBUG', flags_debug)
        emit('CFLAGS_RELEASE', flags_release)
        emit('MASMFLAGS', '')
        emit('DEBUG_INFO_FLAGS', debug_info_flags)
        append('C_WARNING_OPTS', c_warnings)
        append('CXX_WARNING_OPTS', cxx_warnings)

        if self.build.is_release:
            emit('CFLAGS_PER_TYPE', '$CFLAGS_RELEASE')
        if self.build.is_debug:
            emit('CFLAGS_PER_TYPE', '$CFLAGS_DEBUG')
        if self.build.is_ide:
            emit('CFLAGS_PER_TYPE', '@[debug|$CFLAGS_DEBUG]@[release|$CFLAGS_RELEASE]')

        append('CFLAGS', flags, flags_msvs_only, '$CFLAGS_PER_TYPE', '$DEBUG_INFO_FLAGS', '$C_WARNING_OPTS', '$C_DEFINES', '$USER_CFLAGS', '$USER_CFLAGS_GLOBAL')
        append('CXXFLAGS', '$CFLAGS', '/std:c++17', '$CXX_WARNING_OPTS', '$USER_CXXFLAGS')
        append('CONLYFLAGS', flags_c_only, '$USER_CONLYFLAGS')

        append('BC_CFLAGS', '$CFLAGS')
        append('BC_CXXFLAGS', '$BC_CFLAGS', '$CXXFLAGS')

        ucrt_include = os.path.join(self.tc.kit_includes, 'ucrt') if not self.tc.ide_msvs else "$(UniversalCRT_IncludePath.Split(';')[0].Replace('\\','/'))"

        append('CFLAGS', '/DY_UCRT_INCLUDE="%s"' % ucrt_include)
        append('CFLAGS', '/DY_MSVC_INCLUDE="%s"' % vc_include)

        emit_big('''
            when ($NO_WSHADOW == "yes") {
                C_WARNING_OPTS += /wd4456 /wd4457
            }
            when ($NO_COMPILER_WARNINGS == "yes") {
                C_WARNING_OPTS = /w
                CXX_WARNING_OPTS =
            }
            when ($NO_OPTIMIZE == "yes") {
                OPTIMIZE = /Od
            }''')

        emit('SFDL_FLAG', ['/E', '/C', '/P', '/Fi$SFDL_TMP_OUT'])
        emit('WERROR_FLAG', '/WX')
        emit('WERROR_MODE', self.tc.werror_mode)

        if not self.tc.under_wine:
            emit('CL_WRAPPER', '${YMAKE_PYTHON}', '${input:"build/scripts/fix_msvc_output.py"}', 'cl')
            emit('ML_WRAPPER', '${YMAKE_PYTHON}', '${input:"build/scripts/fix_msvc_output.py"}', 'ml')
        else:
            emit('CL_WRAPPER')
            emit('ML_WRAPPER')

        emit_big('''
            ### @usage: MSVC_FLAGS([GLOBAL compiler_flag]* compiler_flags)
            ###
            ### Add the specified flags to the compilation line of C/C++files.
            ### Flags apply only if the compiler used is MSVC (cl.exe)"
            macro MSVC_FLAGS(Flags...) {
                CFLAGS($Flags)
            }

            macro _SRC_c_nodeps(SRC, OUTFILE, INC...) {
                 .CMD=${TOOLCHAIN_ENV} ${CL_WRAPPER} ${CXX_COMPILER} /c /Fo${OUTFILE} ${SRC} ${EXTRA_C_FLAGS} ${pre=/I :INC} ${CXXFLAGS} ${hide;kv:"soe"} ${hide;kv:"p CC"} ${hide;kv:"pc yellow"}
            }

            macro _SRC_cpp(SRC, SRCFLAGS...) {
                .CMD=${TOOLCHAIN_ENV} ${CL_WRAPPER} ${CXX_COMPILER} /c /Fo$_COMPILE_OUTPUTS ${input;msvs_source:SRC} ${EXTRA_C_FLAGS} ${pre=/I :_C__INCLUDE} ${pre=/I :INCLUDE} ${CXXFLAGS} ${SRCFLAGS} ${hide;kv:"soe"} ${hide;kv:"p CC"} ${hide;kv:"pc yellow"}
            }

            macro _SRC_c(SRC, SRCFLAGS...) {
                .CMD=${TOOLCHAIN_ENV} ${CL_WRAPPER} ${C_COMPILER} /c /Fo$_COMPILE_OUTPUTS ${input;msvs_source:SRC} ${EXTRA_C_FLAGS} ${pre=/I :_C__INCLUDE} ${pre=/I :INCLUDE} ${CFLAGS} ${CONLYFLAGS} ${SRCFLAGS} ${hide;kv:"soe"} ${hide;kv:"p CC"} ${hide;kv:"pc yellow"}
            }

            macro _SRC_m(SRC, SRCFLAGS...) {
            }

            macro _SRC_masm(SRC, SRCFLAGS...) {
                .CMD=${cwd:ARCADIA_BUILD_ROOT} ${TOOLCHAIN_ENV} ${ML_WRAPPER} ${MASM_COMPILER} ${MASMFLAGS} ${SRCFLAGS} ''' + masm_io + ''' ${kv;hide:"p AS"} ${kv;hide:"pc yellow"}
            }''')


class MSVCLinker(MSVC, Linker):
    def __init__(self, tc, build):
        MSVC.__init__(self, tc, build)
        Linker.__init__(self, tc, build)

    def print_linker(self):
        super(MSVCLinker, self).print_linker()

        target = self.build.target

        linker = self.tc.link
        linker_lib = self.tc.lib

        arch = select(no_default=True, selectors=(
            (target.is_x86, 'x86'),
            (target.is_x86_64, 'x64'),
            (target.is_armv7, 'arm'),
        ))

        libpaths = []
        if not self.tc.ide_msvs:
            if self.tc.kit_libs:
                libpaths.extend([os.path.join(self.tc.kit_libs, name, arch) for name in ('um', 'ucrt')])
            libpaths.append(os.path.join(self.tc.vc_root, 'lib', arch))

        ignored_errors = [
            4221
        ]

        flag_machine = '/MACHINE:{}'.format(arch.upper())

        flags_ignore = ['/IGNORE:{}'.format(code) for code in ignored_errors]

        flags_common = ['/NOLOGO', '/ERRORREPORT:PROMPT', '/SUBSYSTEM:CONSOLE', '/TLBID:1', '$MSVC_DYNAMICBASE', '/NXCOMPAT']
        flags_common += flags_ignore
        flags_common += [flag_machine]

        flags_debug_only = []
        flags_release_only = []

        if self.tc.ide_msvs:
            flags_common += ['/INCREMENTAL']
        else:
            flags_common += ['/INCREMENTAL:NO']

        # TODO(nslus): DEVTOOLS-1868 remove restriction.
        if not self.tc.under_wine:
            if self.tc.ide_msvs:
                flags_debug_only.append('/DEBUG:FASTLINK' if not self.tc.use_clang else '/DEBUG')
                flags_release_only.append('/DEBUG')
            else:
                # No FASTLINK for ya make, because resulting PDB would require .obj files (build_root's) to persist
                flags_common.append('/DEBUG')

        if not self.tc.ide_msvs:
            flags_common += ['/LIBPATH:"{}"'.format(path) for path in libpaths]

        link_flags_debug = flags_common + flags_debug_only
        link_flags_release = flags_common + flags_release_only
        link_flags_lib = flags_ignore + [flag_machine]

        stdlibs = [
            'kernel32.lib',
            'user32.lib',
            'gdi32.lib',
            'winspool.lib',
            'shell32.lib',
            'ole32.lib',
            'oleaut32.lib',
            'uuid.lib',
            'comdlg32.lib',
            'advapi32.lib',
            'crypt32.lib',
        ]

        emit('LINK_LIB_CMD', linker_lib)
        emit('LINK_EXE_CMD', linker)
        emit('LINK_LIB_FLAGS', link_flags_lib)
        emit('LINK_EXE_FLAGS_RELEASE', link_flags_release)
        emit('LINK_EXE_FLAGS_DEBUG', link_flags_debug)
        emit('LINK_STDLIBS', stdlibs)
        emit('LDFLAGS_GLOBAL', '')
        emit('LDFLAGS', '')
        emit('OBJADDE', '')

        if self.build.is_release:
            emit('LINK_EXE_FLAGS_PER_TYPE', '$LINK_EXE_FLAGS_RELEASE')
        if self.build.is_debug:
            emit('LINK_EXE_FLAGS_PER_TYPE', '$LINK_EXE_FLAGS_DEBUG')
        if self.build.is_ide and self.tc.ide_msvs:
            emit('LINK_EXE_FLAGS_PER_TYPE', '@[debug|$LINK_EXE_FLAGS_DEBUG]@[release|$LINK_EXE_FLAGS_RELEASE]')

        emit('LINK_EXE_FLAGS', '$LINK_EXE_FLAGS_PER_TYPE')

        emit('LINK_IMPLIB_VALUE')

        # TODO(nslus): DEVTOOLS-1868 remove restriction.
        if self.tc.under_wine:
            emit('LINK_EXTRA_OUTPUT')
            emit('LINK_IMPLIB')
        else:
            emit('LINK_IMPLIB', '/IMPLIB:${output;noext;rootrel:REALPRJNAME.lib}')
            emit('LINK_EXTRA_OUTPUT', '/PDB:${output;noext;rootrel:REALPRJNAME.pdb}')

        if not self.tc.under_wine:
            emit('LIB_WRAPPER', '${YMAKE_PYTHON}', '${input:"build/scripts/fix_msvc_output.py"}', 'lib')
            emit('LINK_WRAPPER', '${YMAKE_PYTHON}', '${input:"build/scripts/fix_msvc_output.py"}', 'link')
        else:
            emit('LIB_WRAPPER')
            emit('LINK_WRAPPER')

        emit('LINK_WRAPPER_DYNLIB', '${YMAKE_PYTHON}', '${input:"build/scripts/link_dyn_lib.py"}', '--arch', 'WINDOWS', '--target', '$TARGET')
        emit('EXPORTS_VALUE')

        emit("GENERATE_MF", '$YMAKE_PYTHON ${input:"build/scripts/generate_mf.py"}',
             '--build-root $ARCADIA_BUILD_ROOT --module-name $REALPRJNAME -o ${output;rootrel;pre=$MODULE_PREFIX;suf=$MODULE_SUFFIX.mf:REALPRJNAME}',
             '-t $MODULE_TYPE $NO_GPL_FLAG -Ya,lics $LICENSE_NAMES -Ya,peers ${rootrel:PEERS}',
             )

        emit_big('''
            when ($EXPORTS_FILE) {
                LINK_IMPLIB_VALUE=$LINK_IMPLIB
                EXPORTS_VALUE=/DEF:${input:EXPORTS_FILE}
            }

            LINK_LIB=${GENERATE_MF} && ${TOOLCHAIN_ENV} ${cwd:ARCADIA_BUILD_ROOT} ${LIB_WRAPPER} ${LINK_LIB_CMD} /OUT:${qe;rootrel:TARGET} \
            ${qe;rootrel:AUTO_INPUT} $LINK_LIB_FLAGS ${hide;kv:"soe"} ${hide;kv:"p AR"} ${hide;kv:"pc light-red"}

            LINK_EXE=${GENERATE_MF} && $GENERATE_VCS_C_INFO_NODEP && ${TOOLCHAIN_ENV} ${cwd:ARCADIA_BUILD_ROOT} ${LINK_WRAPPER} ${LINK_EXE_CMD} /OUT:${qe;rootrel:TARGET} \
            ${LINK_EXTRA_OUTPUT} ${qe;rootrel:SRCS_GLOBAL} ${VCS_C_OBJ_RR} ${qe;rootrel:AUTO_INPUT} $LINK_EXE_FLAGS $LINK_STDLIBS $LDFLAGS $LDFLAGS_GLOBAL $OBJADDE \
            ${qe;rootrel:PEERS} ${hide;kv:"soe"} ${hide;kv:"p LD"} ${hide;kv:"pc blue"}

            LINK_DYN_LIB=${GENERATE_MF} && $GENERATE_VCS_C_INFO_NODEP && ${TOOLCHAIN_ENV} ${cwd:ARCADIA_BUILD_ROOT} ${LINK_WRAPPER} ${LINK_WRAPPER_DYNLIB} ${LINK_EXE_CMD} \
            ${LINK_IMPLIB_VALUE} /DLL /OUT:${qe;rootrel:TARGET} ${LINK_EXTRA_OUTPUT} ${EXPORTS_VALUE} \
            ${qe;rootrel:SRCS_GLOBAL} ${VCS_C_OBJ_RR} ${qe;rootrel:AUTO_INPUT} ${qe;rootrel:PEERS} \
            $LINK_EXE_FLAGS $LINK_STDLIBS $LDFLAGS $LDFLAGS_GLOBAL $OBJADDE ${hide;kv:"soe"} ${hide;kv:"p LD"} ${hide;kv:"pc blue"}

            LINK_EXEC_DYN_LIB=${GENERATE_MF} && $GENERATE_VCS_C_INFO_NODEP && ${TOOLCHAIN_ENV} ${cwd:ARCADIA_BUILD_ROOT} ${LINK_WRAPPER} ${LINK_WRAPPER_DYNLIB} ${LINK_EXE_CMD} \
            /OUT:${qe;rootrel:TARGET} ${LINK_EXTRA_OUTPUT} ${EXPORTS_VALUE} \
            ${qe;rootrel:SRCS_GLOBAL} ${VCS_C_OBJ_RR} ${qe;rootrel:AUTO_INPUT} ${qe;rootrel:PEERS} \
            $LINK_EXE_FLAGS $LINK_STDLIBS $LDFLAGS $LDFLAGS_GLOBAL $OBJADDE ${hide;kv:"soe"} ${hide;kv:"p LD"} ${hide;kv:"pc blue"}

            LINK_FAT_OBJECT=${GENERATE_MF} && $GENERATE_VCS_C_INFO_NODEP && $YMAKE_PYTHON ${input:"build/scripts/touch.py"} $TARGET ${kv;hide:"p LD"} ${kv;hide:"pc light-blue"} ${kv;hide:"show_out"}''')


# TODO(somov): Rename!
Compilers = {
    'gnu': (GnuToolchain, GnuCompiler, LD),
    'clang': (GnuToolchain, GnuCompiler, LD),
    'msvc': (MSVCToolchain, MSVCCompiler, MSVCLinker),
}


class Ragel(object):
    def __init__(self):
        self.rlgen_flags = []
        self.ragel_flags = []
        self.ragel6_flags = []

    def configure_toolchain(self, build, compiler):
        if isinstance(compiler, MSVCCompiler):
            self.set_default_flags(optimized=False)
        elif isinstance(compiler, GnuCompiler):
            self.set_default_flags(optimized=build.is_release and not build.is_sanitized)
        else:
            raise ConfigureError('Unexpected compiler {}'.format(compiler))

    def set_default_flags(self, optimized):
        if optimized:
            self.rlgen_flags.append('-G2')
            self.ragel6_flags.append('-CG2')
        else:
            self.rlgen_flags.append('-T0')
            self.ragel6_flags.append('-CT0')

    def print_variables(self):
        emit('RLGEN_FLAGS', self.rlgen_flags)
        emit('RAGEL_FLAGS', self.ragel_flags)
        emit('RAGEL6_FLAGS', self.ragel6_flags)


class Python(object):
    def __init__(self, tc):
        self.python = None
        self.flags = None
        self.ldflags = None
        self.libraries = None
        self.includes = None
        self.tc = tc

    def configure_posix(self, python=None, python_config=None):
        python = python or preset('PYTHON_BIN') or which('python')
        python_config = python_config or preset('PYTHON_CONFIG') or which('python-config')

        if python is None or python_config is None:
            return

        # python-config dumps each option on one line in the specified order
        config = get_stdout([python_config, '--cflags', '--ldflags', '--includes']) or ''
        config = config.split('\n')
        if len(config) < 3:
            return

        self.python = python
        self.flags = config[0]
        self.ldflags = config[1]
        self.includes = config[2]
        # Do not split libraries from ldflags.
        # They are not used separately and get overriden together, so it is safe.
        # TODO(somov): Удалить эту переменную и PYTHON_LIBRARIES из makelist-ов.
        self.libraries = ''
        if preset('USE_ARCADIA_PYTHON') == 'no' and not preset('USE_SYSTEM_PYTHON') and not self.tc.os_sdk_local:
            raise Exception("Use fixed python (see https://clubs.at.yandex-team.ru/arcadia/15392) or set OS_SDK=local flag")

    def print_variables(self):
        variables = Variables({
            'PYTHON_BIN': self.python,
            'PYTHON_FLAGS': self.flags,
            'PYTHON_LDFLAGS': self.ldflags,
            'PYTHON_LIBRARIES': self.libraries,
            'PYTHON_INCLUDE': self.includes
        })

        variables.update_from_presets()
        variables.reset_if_any(reset_value='PYTHON-NOT-FOUND')
        variables.emit()


class Perl(object):
    # Parse (key, value) from "version='5.26.0';" lines
    PERL_CONFIG_RE = re.compile(r"^(?P<key>\w+)='(?P<value>.*)';$", re.MULTILINE)

    def __init__(self):
        self.perl = None
        self.version = None
        self.privlib = None
        self.archlib = None

    def configure_local(self, perl=None):
        self.perl = perl or preset('PERL') or which('perl')
        if self.perl is None:
            return

        # noinspection PyTypeChecker
        config = dict(self._iter_config(['version', 'privlibexp', 'archlibexp']))
        self.version = config.get('version')
        self.privlib = config.get('privlibexp')
        self.archlib = config.get('archlibexp')

    def print_variables(self, prefix=''):
        variables = Variables({
            prefix + 'PERL': self.perl,
            prefix + 'PERL_VERSION': self.version,
            prefix + 'PERL_PRIVLIB': self.privlib,
            prefix + 'PERL_ARCHLIB': self.archlib,
        })

        variables.reset_if_any(reset_value='PERL-NOT-FOUND')
        variables.emit()

    def _iter_config(self, config_keys):
        # Run perl -V:version -V:etc...
        perl_config = [self.perl] + ['-V:{}'.format(key) for key in config_keys]
        config = get_stdout(perl_config) or ''

        start = 0
        while True:
            match = Perl.PERL_CONFIG_RE.search(config, start)
            if match is None:
                break
            yield match.group('key', 'value')
            start = match.end()


class Setting(object):
    def __init__(self, key, auto=None, convert=None):
        self.key = key

        self.auto = auto
        self.convert = convert

        self.preset = preset(key)
        self.from_user = self.preset is not None

        self._value = Setting.no_value

    @property
    def value(self):
        if self._value is Setting.no_value:
            self._value = self.calculate_value()
        return self._value

    def calculate_value(self):
        if not self.from_user:
            return self.auto if not callable(self.auto) else self.auto()
        else:
            return self.preset if not self.convert else self.convert(self.preset)

    @value.setter
    def value(self, value):
        if self.from_user:
            raise ConfigureError("Variable {key} already set by user to {old}. Can not change it's value to {new}".format(key=self.key, old=self._value, new=value))
        self._value = value

    def emit(self):
        if not self.from_user:
            emit(self.key, self.value)

    no_value = object()


class Cuda(object):
    def __init__(self, build):
        """
        :type build: Build
        """
        self.build = build

        self.have_cuda = Setting('HAVE_CUDA', auto=self.auto_have_cuda, convert=to_bool)

        self.cuda_root = Setting('CUDA_ROOT')
        self.cuda_version = Setting('CUDA_VERSION', auto=self.auto_cuda_version)
        self.use_arcadia_cuda = Setting('USE_ARCADIA_CUDA', auto=self.auto_use_arcadia_cuda, convert=to_bool)
        self.use_arcadia_cuda_host_compiler = Setting('USE_ARCADIA_CUDA_HOST_COMPILER', auto=self.auto_use_arcadia_cuda_host_compiler, convert=to_bool)
        self.cuda_use_clang = Setting('CUDA_USE_CLANG', auto=False, convert=to_bool)
        self.cuda_host_compiler = Setting('CUDA_HOST_COMPILER', auto=self.auto_cuda_host_compiler)
        self.cuda_host_compiler_env = Setting('CUDA_HOST_COMPILER_ENV')
        self.cuda_host_msvc_version = Setting('CUDA_HOST_MSVC_VERSION')
        self.cuda_nvcc_flags = Setting('CUDA_NVCC_FLAGS', auto=[])
        self.cuda_arcadia_includes = Setting('CUDA_ARCADIA_INCLUDES', auto=self.auto_cuda_arcadia_includes, convert=to_bool)

        self.peerdirs = ['build/platform/cuda']

        self.cuda_version_list = map(int, self.cuda_version.value.split('.')) if self.cuda_version.value else None

        self.nvcc_flags = ['-std=c++14' if self.cuda_version_list >= [9, 0] else '-std=c++11']

        if not self.have_cuda.value:
            return

        if self.cuda_host_compiler.value:
            self.nvcc_flags.append('--compiler-bindir=$CUDA_HOST_COMPILER')

        if self.use_arcadia_cuda.value:
            self.cuda_root.value = '$CUDA_RESOURCE_GLOBAL'

            if self.build.target.is_linux_x86_64 and self.build.tc.is_clang:
                # TODO(somov): Эта настройка должна приезжать сюда автоматически из другого места
                self.nvcc_flags.append('-I$OS_SDK_ROOT/usr/include/x86_64-linux-gnu')

    def print_(self):
        self.print_variables()
        self.print_macros()

    def print_variables(self):
        self.have_cuda.emit()
        if not self.have_cuda.value:
            return

        if self.use_arcadia_cuda.value and self.cuda_host_compiler.value is None:
            logger.warning('$USE_ARCADIA_CUDA is set, but no $CUDA_HOST_COMPILER')

        self.setup_vc_root()

        self.cuda_root.emit()
        self.cuda_version.emit()
        self.use_arcadia_cuda.emit()
        self.use_arcadia_cuda_host_compiler.emit()
        self.cuda_use_clang.emit()
        self.cuda_host_compiler.emit()
        self.cuda_host_compiler_env.emit()
        self.cuda_host_msvc_version.emit()
        self.cuda_nvcc_flags.emit()

        emit('NVCC_UNQUOTED', self.build.host.exe('$CUDA_ROOT', 'bin', 'nvcc'))
        emit('NVCC', '${quo:NVCC_UNQUOTED}')
        emit('NVCC_FLAGS', self.nvcc_flags, '$CUDA_NVCC_FLAGS')
        emit('NVCC_OBJ_EXT', '.o' if not self.build.target.is_windows else '.obj')

    def print_macros(self):
        cmd_vars = {
            'skip_nocxxinc': '' if self.cuda_arcadia_includes.value else '--y_skip_nocxxinc',
            'includes': '${pre=-I:_C__INCLUDE} ${pre=-I:INCLUDE}' if self.cuda_arcadia_includes.value else '-I$ARCADIA_ROOT',
        }

        if not self.cuda_use_clang.value:
            cmd = '$YMAKE_PYTHON ${input:"build/scripts/compile_cuda.py"} ${tool:"tools/mtime0"} $NVCC $NVCC_FLAGS -c ${input:SRC} -o ${output;suf=${OBJ_SUF}${NVCC_OBJ_EXT}:SRC} %(skip_nocxxinc)s %(includes)s --cflags $C_FLAGS_PLATFORM $CFLAGS $SRCFLAGS ${input;hide:"build/platform/cuda/cuda_runtime_include.h"} $CUDA_HOST_COMPILER_ENV ${kv;hide:"p CC"} ${kv;hide:"pc light-green"}'
        else:
            cmd = '$CXX_COMPILER --cuda-path=$CUDA_ROOT $C_FLAGS_PLATFORM -c ${input:SRC} -o ${output;suf=${OBJ_SUF}${NVCC_OBJ_EXT}:SRC} %(includes)s $CXXFLAGS $SRCFLAGS $TOOLCHAIN_ENV ${kv;hide:"p CU"} ${kv;hide:"pc green"}'

        emit_big('''
            macro _SRC("cu", SRC, SRCFLAGS...) {
                .CMD=%s
                .PEERDIR=%s
            }
        ''' % (cmd % cmd_vars, ' '.join(sorted(self.peerdirs))))

    def have_cuda_in_arcadia(self):
        host, target = self.build.host_target

        if not host.is_linux_x86_64 and not host.is_macos_x86_64 and not host.is_windows_x86_64:
            return False

        # We have no CUDA cross-build yet
        if host != target:
            return False

        if self.cuda_version.value in ('9.0', '9.1', '9.2', '10.0', '10.1'):
            return True

        if self.cuda_version.value == '8.0' and host.is_linux_x86_64:
            return True

        return False

    def auto_have_cuda(self):
        if self.build.is_sanitized:
            return False
        return self.cuda_root.from_user or self.use_arcadia_cuda.value and self.have_cuda_in_arcadia()

    def auto_cuda_version(self):
        if self.use_arcadia_cuda.value:
            return '10.1'

        if not self.have_cuda.value:
            return None

        nvcc_exe = self.build.host.exe(os.path.expanduser(self.cuda_root.value), 'bin', 'nvcc')

        def error():
            raise ConfigureError('Failed to get CUDA version from {}'.format(nvcc_exe))

        version_output = get_stdout([nvcc_exe, '--version']) or error()
        match = re.search(r'^Cuda compilation tools, release (\d+\.\d+),', version_output, re.MULTILINE) or error()

        return match.group(1)

    def auto_use_arcadia_cuda(self):
        return not self.cuda_root.from_user

    def auto_use_arcadia_cuda_host_compiler(self):
        return not self.cuda_host_compiler.from_user and not self.cuda_use_clang.value

    def auto_cuda_host_compiler(self):
        if not self.use_arcadia_cuda_host_compiler.value:
            return None

        host, target = self.build.host_target

        if host.is_windows_x86_64 and target.is_windows_x86_64:
            return self.cuda_windows_host_compiler()

        return select((
            (host.is_linux_x86_64 and target.is_linux_x86_64, '$CUDA_HOST_TOOLCHAIN_RESOURCE_GLOBAL/bin/clang'),
            (host.is_macos_x86_64 and target.is_macos_x86_64, '$CUDA_HOST_TOOLCHAIN_RESOURCE_GLOBAL/usr/bin/clang'),
        ))

    def cuda_windows_host_compiler(self):
        vc_version = {
            '10.1': '14.13.26128',  # (not latest)
            '10.0': '14.13.26128',  # (not latest)
            '9.2': '14.13.26128',
            '9.1': '14.11.25503',
            '9.0': '14.11.25503',
        }[self.cuda_version.value]

        env = {
            'Y_VC_Version': vc_version,
            'Y_SDK_Version': self.build.tc.sdk_version,
            'Y_SDK_Root': '$WINDOWS_KITS_RESOURCE_GLOBAL',
        }
        env['Y_VC_Root'] = '$CUDA_HOST_TOOLCHAIN_RESOURCE_GLOBAL/VC/Tools/MSVC/%(Y_VC_Version)s' % env

        if not self.build.tc.ide_msvs:
            self.peerdirs.append('build/platform/msvc')
        self.cuda_host_compiler_env.value = format_env(env)
        self.cuda_host_msvc_version.value = vc_version
        return '%(Y_VC_Root)s/bin/HostX64/x64/cl.exe' % env

    def setup_vc_root(self):
        if not self.cuda_host_compiler.from_user:
            return  # Already set in cuda_windows_host_compiler()

        if self.cuda_host_compiler_env.from_user:
            return  # We won't override user setting

        def is_root(dir):
            return all(os.path.isdir(os.path.join(dir, name)) for name in ('bin', 'include', 'lib'))

        def get_root():
            path, old_path = os.path.normpath(self.cuda_host_compiler.value), None
            while path != old_path:
                if is_root(path):
                    return path
                path, old_path = os.path.dirname(path), path

        vc_root = get_root()
        if vc_root:
            self.cuda_host_compiler_env.value = format_env({'Y_VC_Root': vc_root})

    def auto_cuda_arcadia_includes(self):
        return self.cuda_version_list >= [9, 0]


class Yasm(object):
    def __init__(self, target):
        self.yasm_tool = '${tool:"contrib/tools/yasm"}'
        self.fmt = None
        self.platform = None
        self.target = target
        self.flags = []

    def configure(self):
        if self.target.is_ios or self.target.is_macos:
            self.platform = ['DARWIN', 'UNIX']
            self.fmt = 'macho'
        elif (self.target.is_windows and self.target.is_64_bit) or self.target.is_cygwin:
            self.platform = ['WIN64']
            self.fmt = 'win'
        elif self.target.is_windows and self.target.is_32_bit:
            self.platform = ['WIN32']
            self.fmt = 'win'
        else:
            self.platform = ['UNIX']
            self.fmt = 'elf'

        if self.fmt == 'elf':
            self.flags += ['-g', 'dwarf2']

    def print_variables(self):
        d_platform = ' '.join([('-D ' + i) for i in self.platform])
        output = '${{output;noext;suf={}:SRC}}'.format('${OBJ_SUF}.o' if self.fmt != 'win' else '${OBJ_SUF}.obj')
        print '''\
macro _SRC_yasm_impl(SRC, PREINCLUDES[], SRCFLAGS...) {{
    .CMD={} -f {}$HARDWARE_ARCH {} $YASM_DEBUG_INFO_DISABLE_CACHE__NO_UID__ -D ${{pre=_;suf=_:HARDWARE_TYPE}} -D_YASM_ $ASM_PREFIX_VALUE {} ${{YASM_FLAGS}} ${{pre=-I :_ASM__INCLUDE}} ${{pre=-I :INCLUDE}} ${{SRCFLAGS}} -o {} ${{pre=-P :PREINCLUDES}} ${{input;hide:PREINCLUDES}} ${{input:SRC}} ${{kv;hide:"p AS"}} ${{kv;hide:"pc light-green"}}

}}
'''.format(self.yasm_tool, self.fmt, d_platform, ' '.join(self.flags), output)


def main():
    options = opts()

    arcadia = Arcadia(options.arcadia_root)

    ymake = YMake(arcadia)

    ymake.print_core_conf()
    ymake.print_presets()
    ymake.print_settings()

    build = Build(arcadia, options.build_type, options.toolchain_params, force_ignore_local_files=not options.local_distbuild)
    build.print_build()

    emit('CONF_SCRIPT_DEPENDS', __file__)


if __name__ == '__main__':
    main()
