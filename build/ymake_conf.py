#!/usr/bin/env python
# main ymake configuration script:
# defines platform-dependent variables and command patterns

import os
import sys
import platform
import re
import logging
import subprocess
import optparse
import json
import base64


DIST_PREFIX = "dist-"
VERBOSE_KEY = "verbose"
CORE_CONF_FILE = "ymake.core.conf"
IDE_BUILD_TYPE = "nobuild"


class Result(object):
    pass


class FailedCmdException(Exception):
    pass


def lazy(func):
    result = Result()

    def wrapper():
        try:
            return result.result
        except AttributeError:
            result.result = func()

        return result.result

    return wrapper


def init_logger(verbose=False):
    logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO)


def which(prog):
    if os.path.exists(prog) and os.access(prog, os.X_OK):
        return prog
    path = os.getenv('PATH', '')
    for p in path.split(os.path.pathsep):
        p = os.path.join(p, prog)
        if os.path.exists(p) and os.path.isfile(p) and os.access(p, os.X_OK):
            return p


def version_as_list(v):
    return list(map(int, (v.split("."))))


def convert_version(v):
    return int(''.join([x.zfill(2) for x in v.split('.')]))


def system_command_call(command, verbose, env=None):
    try:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env)
        stdout, stderr = process.communicate()
        if process.returncode != 0:
            if verbose:
                logging.error('"%s" returned %s. Stdout: "%s", Stderr: "%s"',
                              ' '.join(command), process.returncode, stdout, stderr)
            raise FailedCmdException('{0} failed with exit code {1}.'.format(' '.join(command), process.returncode))
        return stdout, stderr
    except OSError:
        return None


def get_stdout(command, verbose=True):
    sys_call_result = system_command_call(command, verbose)
    if sys_call_result:
        return sys_call_result[0][:-1]
    return None


def get_stderr(command, verbose=True, env=None):
    sys_call_result = system_command_call(command, verbose, env)
    if sys_call_result:
        return sys_call_result[1][:-1]
    return None


def xstr(s):
    return str(s) if s else ''


def flatten(l):
    return sum(map(flatten, l), []) if isinstance(l, list) or isinstance(l, tuple) else [l]


def emit(key, *value):
    print '{0}={1}'.format(key, ' '.join(map(xstr, flatten(value))))


def append(key, *value):
    print '{0}+={1}'.format(key, ' '.join(map(xstr, flatten(value))))


# TODO: replace with ${extfile:}
def quote_path(v):
    if v is None:
        return None
    if ' ' in v:
        return "\"{0}\"".format(v)
    return v


class BuildTypeBlock(object):
    def __init__(self, name, *parents):
        self.name = name
        self._parents = parents
        self._inner = []
        self._subblocks = []

    def emit(self, key, *value):
        return self.add('{0}={1}'.format(key, ' '.join(map(xstr, flatten(value)))))

    def append(self, key, *value):
        return self.add('{0}+={1}'.format(key, ' '.join(map(xstr, flatten(value)))))

    def add(self, text):
        self._inner.append(text)
        return self

    def sub_block(self, name):
        self._subblocks.append(BuildTypeBlock(name))
        return self._subblocks[-1]

    def __write(self, stream, padding):
        stream.write('#' + padding + 'BuildType  ' + self.name)
        if self._parents:
            stream.write(' :  ' + ', '.join(map(xstr, flatten(self._parents))) + ' ')
        stream.write(' {\n')

        for x in self._inner:
            stream.write(padding + '  ' + x + '\n')

        for x in self._subblocks:
            x.__write(stream, padding + '  ')

        stream.write('#' + padding + '}\n\n')

    def write(self, stream=sys.stdout):
        self.__write(stream, padding='')
        return self


def print_conf_style(blk, conf_dict):
    for key in conf_dict:
        val = conf_dict[key]
        if key == "YMAKE_PRINT_SPECS" and val:
            print val
        else:
            blk.emit(key, val)


def to_yes_no(bool_var):
    if bool_var == "True":
        return "yes"
    if bool_var == "False":
        return "no"
    if bool_var == "None":
        return None
    return bool_var


def str2bool(s):
    return str(s).lower() in ['true', 'yes', 'on']


def userify_presets(presets, keys):
    for key in keys:
        user_key = 'USER_{}'.format(key)
        if key not in presets:
            presets[user_key] = ''
            continue
        value = presets.pop(key)
        if user_key in presets:
            value += ' ' + presets[user_key]
        presets[user_key] = value


@lazy
def load_presets():
    return opts().presets


def preset(key):
    presets = load_presets()
    if key in presets:
        logging.debug("Use preset of {0} -> {1}".format(key, presets[key]))
        return presets[key]
    return None


def get_by_func(lst, func):
    for x in lst:
        value = func(x)
        if value is not None:
            return value
    return None


def get_from_var(alias):
    return get_by_func(alias, preset) or get_by_func(alias, os.getenv)


def get_from_params_or_preset(name):
    try:
        params = load_toolchain_params()
        return params[name.lower()]
    except (KeyError, ValueError, TypeError):
        return preset(name.upper()) or name.lower()


def preset_has_value(key, value):
    val = preset(key)
    if val and value in val:
        return True
    return False


def is_positive(key):
    return preset_has_value(key, 'yes')


def is_negative(key):
    return preset_has_value(key, 'no')


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
        opt_group.add_option('-D', '--preset', dest='presets', action='append', default=[],
                             help='set or override presets')
        opt_group.add_option('-l', '--local-distbuild', dest='local_distbuild', action='store_true', default=False,
                             help='conf for local distbuild')
        parser.add_option_group(opt_group)

        self.options, self.arguments = parser.parse_args(argv)

        argv = self.arguments
        if len(argv) < 4:
            print >> sys.stderr, "Usage: ArcRoot, --BuildType--, Verbosity, [Path to local.ymake]"
            sys.exit(1)

        self.build_type = argv[2].lower()

        self.resolve_tc_pathes = True  # true for ya build and ymake binary
        self.mine_system_pathes = True  # true for local build

        self.build_system = "ymake"  # XXX: deprecated, remove ASAP

        if self.build_type.startswith(DIST_PREFIX):
            self.build_system = "distbuild"
            self.build_type = self.build_type[len(DIST_PREFIX):]
            self.mine_system_pathes = self.options.local_distbuild
            self.resolve_tc_pathes = False
        else:
            if self.build_type == IDE_BUILD_TYPE:
                self.resolve_tc_pathes = False
                self.mine_system_pathes = False
            if self.options.local_distbuild:
                logging.warn("Not applicable for non-dist build: --local-distbuild key")  # XXX: Remove ASAP when refactor ymake

        init_logger(argv[3] == VERBOSE_KEY)

        for attr, val in self.options.__dict__.items():
            setattr(self, attr, val)

        self.presets = parse_presets(self.presets)
        userify_presets(self.presets, ('CFLAGS', 'CXXFLAGS', 'CONLYFLAGS'))


@lazy
def opts():
    return Options(sys.argv)


@lazy
def script_args():
    return opts().arguments


@lazy
def conf_platform():
    sys_os = platform.system().lower()
    return sys_os


class YMake(object):
    def __init__(self, argv):
        self.arc_root = argv[1]

    @staticmethod
    def find_core_conf():
        script_dir = os.path.dirname(__file__)
        full_path = os.path.join(script_dir, CORE_CONF_FILE)
        if os.path.exists(full_path):
            return full_path
        return None

    def print_settings(self):
        emit("ARCADIA_ROOT", self.arc_root)
        emit("CURRENT_STLPORT_VERSION", "5.1.4")
        emit("SWIG_LANG", preset("SWIG_LANG") or "python")
        emit("BYACC_FLAGS", preset("BYACC_FLAGS") or "-v")
        emit("BISON_FLAGS", preset("BISON_FLAGS") or "-v")
        emit("GP_FLAGS", preset("GP_FLAGS") or "-CtTLANSI-C -Dk* -c")

        # TODO: move the following stuff to ymake.core.conf
        emit("CHECKFLAG")
        emit("LEX_FLAGS")
        emit('NO_MAPREDUCE')
        emit('ARCADIA_TEST_ROOT', '../arcadia_tests_data/')
        print """
when ($NO_MAPREDUCE ==  \"yes\") {
    C_DEFINES += -DNO_MAPREDUCE
}
"""

    @staticmethod
    def print_build_type(build_type):
        blk = BuildTypeBlock("name=YMake.bt={0}".format(build_type))
        blk.emit("BUILD_TYPE", build_type)
        blk.emit('BT_' + build_type.upper().replace('-', '_'), 'yes')
        if build_type == 'valgrind' or build_type == 'valgrind-release':
            blk.emit('WITH_VALGRIND', 'yes')
        blk.write()

    @staticmethod
    def print_build_system(build_system):
        blk = BuildTypeBlock("name=YMake.bs={0}".format(build_system))
        unpickler = "$ARCADIA_ROOT/build/plugins/_unpickler.py"
        if build_system == "distbuild":
            sub_blk = blk.sub_block("bs=distbuild")
            sub_blk.emit("DISTBUILD", "yes")
            if is_positive("NO_YMAKE"):
                # these systems might not have ymake available
                sub_blk.emit("YMAKE_PYTHON", "$(PYTHON)/python")
                sub_blk.emit("YMAKE_UNPICKLER", "$(PYTHON)/python", unpickler)
            else:
                my_ymake = opts().presets.get('MY_YMAKE_BIN', "$(YMAKE)/ymake")
                sub_blk.emit("YMAKE_PYTHON", my_ymake, "--python")
                sub_blk.emit("YMAKE_UNPICKLER", my_ymake, "--python", unpickler)
        elif build_system == "ymake":
            sub_blk = blk.sub_block("bs=ymake")
            sub_blk.emit("YMAKE_PYTHON", "$YMAKE_BIN", "--python")
            sub_blk.emit("YMAKE_UNPICKLER", "$YMAKE_BIN", "--python", unpickler)
        blk.write()

    @staticmethod
    def print_pic(pic):
        blk = BuildTypeBlock("name=YMake.PIC={0}".format(pic))
        if pic == "yes":
            blk.emit("PIC", "yes")
        blk.write()

@lazy
def ymake():
    return YMake(script_args())


def validate_os_name(os_name):
    if os_name == "WIN" or os_name == "WIN32" or os_name == "WIN64":
        return "WINDOWS"
    if os_name.startswith('CYGWIN'):
        return 'CYGWIN'
    return os_name


class OStypes(object):
    Unix, Win, Unknown = range(3)


class LocalSystem(object):
    def __init__(self):
        self.type = platform.machine()
        if self.type and "amd64" in self.type.lower():
            self.type = "x86_64"
        sys_os = conf_platform().upper()
        self.os = validate_os_name(sys_os)
        self.os_type = OStypes.Unix
        if self.os == "WINDOWS":
            self.os_type = OStypes.Win
        self.version = [None, None]
        if self.os == "FREEBSD":
            p = re.compile("^([0-9]+)\.([0-9]+)")
            m = p.match(platform.release())
            if m:
                self.version[0] = m.group(1)
                self.version[1] = m.group(2)


class System(object):
    def __init__(self, os, type):
        self.os_type = OStypes.Unknown
        self.os = validate_os_name(os)
        self.type = type
        self.arch = 32 if "i386" in type or "i686" in type or 'arm' in type else 64

        if 'arm' in type or 'aarch' in type or 'ppc' in type:
            self.x86 = False
        else:
            self.x86 = True

        os_opts = {
            "WINDOWS": System.init_win,
            "LINUX": System.init_linux,
            "FREEBSD": System.init_freebsd,
            "DARWIN": System.init_darwin,
            "ANDROID": System.init_android,
            "IOS": System.init_darwin,
            "CYGWIN": System.init_cygwin
        }

        if self.os not in os_opts:
            logging.error("Platform {0} is not supported for now!".format(self.os))
            sys.exit(1)

        os_opts[self.os](self)

    def init_cygwin(self):
        self.init_linux()

    def init_linux(self):
        self.os_type = OStypes.Unix
        self.format = 'elf'
        self.yasm_platform = 'UNIX'
        self.env_delim = ':'

    def init_android(self):
        self.os_type = OStypes.Unix
        self.format = 'elf'
        self.yasm_platform = 'UNIX'
        self.env_delim = ':'

    def init_freebsd(self):
        self.os_type = OStypes.Unix
        self.format = 'elf'
        self.yasm_platform = 'UNIX'
        self.env_delim = ':'

    def init_win(self):
        self.os_type = OStypes.Win
        self.env_delim = ';'

    def init_darwin(self):
        self.os_type = OStypes.Unix
        self.format = 'macho'
        self.yasm_platform = 'DARWIN'
        self.env_delim = ':'

    # TODO: rename WIN32 to WINDOWS in CMakeLists, define WIN32 for 32-bit Windows only
    def print_windows_target_const(self, par_blk):
        par_blk.emit("WINDOWS", "yes")
        par_blk.emit("WIN32", "yes")
        if self.arch == 64:
            par_blk.emit("WIN64", "yes")

    @staticmethod
    def print_windows_host_const(par_blk):
        par_blk.emit('CCVERS', '0')

    def print_nix_target_const(self, par_blk):
        par_blk.emit('JAVA_INCLUDE', '-I{0}/include -I{0}/include/{1}'.format('/usr/lib/jvm/default-java', self.os.lower()))

        par_blk.emit("UNIX", "yes")
        par_blk.emit("REALPRJNAME")
        par_blk.emit("SONAME")
        par_blk.emit("USE_CUDA", "no")
        par_blk.emit("CUDA_NVCC_FLAGS", preset("CUDA_NVCC_FLAGS") or "")
        par_blk.emit("NVCCOPTS", "--compiler-options", "-fno-strict-aliasing", "-I${ARCADIA_ROOT}", "-I.",
                     "-I${CUDA_ROOT}/SDK/common/inc", "-I${CUDA_ROOT}/include", "-DUNIX", "-O3", "$CUDA_NVCC_FLAGS")

    @staticmethod
    def print_nix_host_const(par_blk):
        par_blk.emit("WRITE_COMMAND", "/bin/echo", "-e")

        par_blk.add("""
  when ($USE_PYTHON) {
      C_DEFINES+= -DUSE_PYTHON
  }

  when ($USE_CUDA == \"yes\") {
      CFLAGS+= -fno-strict-aliasing -I${CUDA_ROOT}/SDK/common/inc -I${CUDA_ROOT}/include -DUNIX
  }
  CUDART=-lcudart_static
  when ($PIC) {
      CUDART=-lcudart
  }
  when ($USE_CUDA == \"yes\") {
      LINK_OPTIONS_END+= -L${CUDA_ROOT}/lib64 $CUDART
  }
""")

    def print_freebsd_const(self, par_blk):
        if l_system().os == self.os and self.type == l_system().type:
            major = l_system().version[0]
            minor = l_system().version[1]
        else:
            major = '9'
            minor = '0'

        par_blk.add('FREEBSD_VER=%s' % major)
        par_blk.add('FREEBSD_VER_MINOR=%s' % minor)
        par_blk.add("""
  when (($USEMPROF == \"yes\") || ($USE_MPROF == \"yes\")) {
      MPROFLIB+= -L/usr/local/lib -lc_mp
  }
  when (($USEMPROF == \"yes\") || ($USE_MPROF == \"yes\")) {
      C_DEFINES+= -DUSE_MPROF
  }
""")

    @staticmethod
    def print_linux_const(par_blk):
        par_blk.add("""
  when (($USEMPROF == \"yes\") || ($USE_MPROF == \"yes\")) {
      MPROFLIB+= -ldmalloc
  }
""")

    def print_target_settings(self):
        blk = BuildTypeBlock("name=System.tOS={0}.tType={1}".format(self.os, self.type))
        blk.emit("TARGET_PLATFORM", self.os)
        blk.emit("HARDWARE_ARCH", self.arch)
        blk.emit("HARDWARE_TYPE", self.type)

        if self.arch == 32:
            blk.emit("ARCH_TYPE_32", "yes")
        elif self.arch == 64:
            blk.emit("ARCH_TYPE_64", "yes")
        else:
            logging.error("Unsupported ARCH_TYPE: %s", self.arch)

        if "i386" in self.type:
            blk.emit("ARCH_I386", "yes")
        elif "i686" in self.type:
            blk.emit("ARCH_I386", "yes")
            blk.emit("ARCH_I686", "yes")
        elif "amd64" in self.type or 'x86_64' in self.type:
            blk.emit("ARCH_X86_64", "yes")
        elif "arm" in self.type:
            blk.emit("ARCH_ARM", "yes")

            if "arm7" in self.type:
                blk.emit("ARCH_ARM7", "yes")
            if "arm64" in self.type:
                blk.emit("ARCH_ARM64", "yes")
            if "armv8" in self.type:
                blk.emit("ARCH_ARM64", "yes")
        elif "k1om" in self.type:
            blk.emit("MIC_ARCH", "yes")  # TODO: Remove when not used
            blk.emit("ARCH_K1OM", "yes")
        elif "aarch64" in self.type:
            blk.emit("ARCH_AARCH64", "yes")
        elif "ppc64le" == self.type:
            blk.emit("ARCH_PPC64LE", "yes")
        else:
            logging.error("Unsupported ARCH: %s", self.type)

        if self.os:
            blk.emit(self.os, "yes")
            blk.emit("OS_" + self.os, "yes")
        if self.os_type == OStypes.Unix:
            self.print_nix_target_const(blk)
            if self.os == "LINUX":
                self.print_linux_const(blk)
            elif self.os == "FREEBSD":
                self.print_freebsd_const(blk)
            # TODO emit android
        elif self.os_type == OStypes.Win:
            self.print_windows_target_const(blk)
        self.print_target_shortcuts(blk)
        self.target_blk_name = blk.name
        blk.write()

    # Misc target arch-related shortcuts
    def print_target_shortcuts(self, blk):
        if preset('HAVE_CUDA') is None:
            blk.add("HAVE_CUDA=no")
            if self.os == "LINUX":
                blk.add("""
  when ($ARCH_X86_64 && !$SANITIZER_TYPE && !$PIC) {
      HAVE_CUDA=yes
  }
""")
            if self.os in ("LINUX", "ANDROID"):
                blk.add("""
  when ($ARCH_AARCH64 && $ARM_CUDA) {
      HAVE_CUDA=yes
  }
""")

        if preset('HAVE_MKL') is None:
            blk.add("HAVE_MKL=no")
            if self.os == "LINUX":
                blk.add("""
  when ($ARCH_X86_64 && !$SANITIZER_TYPE) {
      HAVE_MKL=yes
  }
""")

    def print_host_settings(self):
        blk = BuildTypeBlock("name=System.bs=*.hOS={0}.hType={1}".format(self.os, self.type))
        blk.emit("HOST_PLATFORM", self.os)
        if self.os_type == OStypes.Unix:
            self.print_nix_host_const(blk)
        elif self.os_type == OStypes.Win:
            self.print_windows_host_const(blk)
        self.host_blk_name = blk.name
        blk.write()


@lazy
def l_system():
    return LocalSystem()


t_systems = {}


def t_system(os, type):
    key = os + '@' + type
    if key not in t_systems:
        t_systems[key] = System(os, type)
    return t_systems[key]

# define C and CXX

cxx_alias = ("CXX", "CXX_COMPILER")
c_alias = ("CC", "C_COMPILER")
arc_profile_alias = ("USE_ARC_PROFILE", )
ar_alias = ("AR", )
werror_mode_alias = ("WERROR_MODE", )


@lazy
def load_toolchain_params():
    return json.loads(base64.b64decode(opts().toolchain_params))


@lazy
def cross_suffix():
    return '' if is_positive('FORCE_NO_PIC') else '.pic'


class Toolchain(object):
    def __init__(self, toolchain_name):
        self.name = toolchain_name
        self.supplied_by_user = False
        self.tool_toolchain = None
        self.werror_mode = None
        self.name_marker = None
        self.toolchain_root = None
        self.toolchain_root_obtained = False
        self.yatc = load_toolchain_params()
        self.params = self.yatc["params"]
        self.env = self.yatc.get("env", {})
        marker = self.params["match_root"] if "match_root" in self.params else self.name.upper()
        self.name_marker = "$(%s)" % marker
        self.type = self.params["type"]

        if self.type == 'system_cxx':
            self.supplied_by_user = True
            self.cxx_compiler = self.params['cxx_compiler']
            self.toolchain_by_cxx_compiler_file_name()
            return

        self.toolchain_root = self.name_marker + '/'

        self.host_platforms = [p['host']['os'] for p in self.yatc.get('platforms', []) if 'host' in p and 'os' in p['host']]
        # TODO: wipe, also WIN64 is not used
        if any(x in self.host_platforms for x in ('WIN', 'WIN64')):
            self.host_platforms.append('WINDOWS')

        self.werror_mode = str(get_from_var(werror_mode_alias) or self.params["werror_mode"])

    # final toolchain name and C compiler
    def toolchain_by_cxx_compiler_file_name(self):
        basic_name = cxx_exec_name = os.path.basename(self.cxx_compiler)
        trailing_ver = ""
        p = re.compile("(.+?)([-.0-9][-+.0-9]*)$")
        m = p.match(cxx_exec_name)
        if m:
            basic_name = m.group(1)
            trailing_ver = m.group(2)

        known_cxx_names = {
            "g++": ["gcc", "gcc", "gnu", Toolchain.detect_gcc],
            "clang++": ["clang", "clang", "clang", Toolchain.detect_gcc],
            "icpc": ["icc", "icc", "intel", Toolchain.detect_gcc],
            "cl.exe": ["cl.exe", "", "msvc", Toolchain.detect_msvc_],
        }
        cxx_info = known_cxx_names.get(basic_name)
        if not cxx_info:
            logging.error("Could not determine compiler type of " + cxx_exec_name + ". Stop build!")
            logging.error("Please inform ymake-dev@ team of your compiler's C and C++ executable names and version string")
            sys.exit(1)
        self.type = cxx_info[2]
        self.c_compiler = os.path.join(os.path.dirname(self.cxx_compiler), cxx_info[0] + trailing_ver)
        self.params["c_compiler"] = self.c_compiler
        self.params["cxx_compiler"] = self.cxx_compiler
        if cxx_info[3] is not None:
            cxx_info[3](self)
        else:
            self.name = cxx_info[1] + trailing_ver

    def detect_gcc(self):
        SystemGCC(self)

    def detect_msvc_(self):
        detect_msvc(self)

    def local_path(self, param):
        if param not in self.params:
            return None
        value = self.params[param]
        return self.convert_path(value)

    def convert_path(self, value):
        # here we assume that prefixes like $(GCC) from config are directly understood by distbuild
        if not opts().resolve_tc_pathes:
            return value
        self.check_toolchain_root()
        if self.toolchain_root is not None and self.name_marker is not None:
            return value.replace(self.name_marker, self.toolchain_root)
        else:
            return value

    # this function's only call is commented out
    def check_toolchain_root(self):
        if self.supplied_by_user or self.toolchain_root_obtained or self.name == "theyknow":
            return

        raise NotImplementedError()


def choose_compiler(build_settings, tc, target_os):
    build_type = build_settings.build_type
    if tc.type == "gnu":
        return GnuGCC(tc, target_os, build_type), LD(tc, target_os, build_type)
    if tc.type == "clang":
        return ClangCC(tc, target_os, build_type), LD(tc, target_os, build_type)
    if tc.type == "intel":
        return ICC(tc, target_os, build_type), ICCLD(tc, target_os, build_type)
    if tc.type == "intel_mic":
        return ICCwithMIC(tc, target_os, build_type), ICCwithMICLD(tc, target_os, build_type)
    if tc.type == "msvc":
        return MSVC(build_settings, tc), MSVCLinker(build_settings, tc)
    return None


t_full_build_types = {}


def print_full_build_type(build_settings, toolchain_name, build_sys, host_os_id, target_os_id, target_type):
    host_os = t_system(host_os_id, target_type)
    host_os.print_host_settings();

    target_os = t_system(target_os_id, target_type)
    target_os.print_target_settings();

    tc = Toolchain(toolchain_name)

    print "# toolchain.type = ", tc.type
    compiler, linker = choose_compiler(build_settings, tc, target_os)

    my_build_type = "name=YMake.bt=*.tc={0}.bs={1}.hOS={2}.tOS={3}.tType={4}.PIC=*".format(tc.name, build_sys, host_os_id, target_os_id, target_type)
    if my_build_type in t_full_build_types:
        return my_build_type
    t_full_build_types[my_build_type] = True

    blk = BuildTypeBlock(my_build_type,
                         "name=YMake.bt=*",
                         "name=YMake.bs=*",
                         "name=YMake.PIC=*",
                         print_other_settings(build_sys, host_os_id),
                         host_os.host_blk_name, target_os.target_blk_name, compiler.parent_blocks)
    blk.emit("COMPILER_ID", tc.type.upper())
    compiler.print_compiler_settings(blk)

    compiler.print_compiler_cmd(blk)

    linker.print_linker_settings(blk)
    linker.print_linker_cmd(blk)

    blk.write()
    res = blk.name
    return res


gcc_sse_opt_names = {"-msse3": "-DSSE3_ENABLED=1", "-msse2": "-DSSE2_ENABLED=1", "-msse": "-DSSE_ENABLED=1"}


class GNU(object):
    gcc_fstack = ["-fstack-protector"]

    def __init__(self, tc, target_os, build_type):
        self.tc = tc
        self.target_os = target_os
        self.c_defines = ["-D_FILE_OFFSET_BITS=64", "-D_LARGEFILE_SOURCE",
                          "-D__STDC_CONSTANT_MACROS", "-D__STDC_FORMAT_MACROS", "-DGNU"]
        if target_os.os in ('LINUX', 'CYGWIN'):
            self.c_defines.append('-D_GNU_SOURCE')
        self.extra_compile_opts = []

        self.verify_compiler()

        self.parent_blocks = [gcc_codegen(tc, target_os.type, "14").name, gcc_build_type(tc, build_type).name]

        def split_preset(name):
            return preset(name).split() if preset(name) else []

        self.gcc_sys_lib = []
        for sys_lib in tc.params.get('sys_lib', []):
            self.gcc_sys_lib.append(tc.convert_path(sys_lib))

        self.tc.params["have_valgrind_headers"] = check_valgrind_headers()
        self.c_flags = []
        self.c_flags.append(self.tc.params['arch_opt'] if 'arch_opt' in self.tc.params else "-m64")
        self.c_flags.append("-pipe")
        self.cxx_flags = []
        self.c_only_flags = []
        self.sfdl_flags = ['-E', '-C', '-x', 'c++']

        if not preset("NOSSE"):
            for opt in self.tc.params.get("supported_codegen_opts", []):
                self.c_flags.append(opt)
                if opt in gcc_sse_opt_names:
                    self.c_defines.append(gcc_sse_opt_names[opt])
        else:
            self.c_defines.append(self.c_defines.append("-no-sse"))

    def print_compiler_settings(self, blk):
        blk.emit("GCC", "yes")
        blk.emit("GCC_VER", self.tc.params["gcc_version"])
        self.print_gnu_compiler_common_settings(blk)

    def print_gnu_compiler_common_settings(self, blk):
        blk.emit("CCVERS", convert_version(self.tc.params["gcc_version"]))
        blk.emit("C_COMPILER_UNQUOTED", self.tc.local_path("c_compiler"))
        blk.emit("C_COMPILER", "${quo:C_COMPILER_UNQUOTED}")
        blk.emit("WERROR_MODE", self.tc.werror_mode or "compiler_specific")
        blk.emit("FSTACK", self.gcc_fstack)
        blk.append("C_DEFINES", self.c_defines, "-D_THREAD_SAFE", "-D_PTHREADS", "-D_REENTRANT")
        blk.emit("DUMP_DEPS")
        blk.emit("GCC_PREPROCESSOR_OPTS", "$DUMP_DEPS", "$C_DEFINES")
        blk.append("C_WARNING_OPTS", "-Wall", "-W", "-Wno-parentheses")
        blk.append("CXX_WARNING_OPTS", "-Woverloaded-virtual")
        blk.append("USER_CFLAGS_GLOBAL", "")
        blk.append("USER_CFLAGS_GLOBAL", "")
        blk.add("""
  when ($PIC && $PIC == \"yes\") {
      PICFLAGS=-fPIC
  }
  otherwise {
      PICFLAGS=
  }
""");
        blk.append("CFLAGS", self.c_flags, '$DEBUG_INFO_FLAGS', '$GCC_PREPROCESSOR_OPTS', '$C_WARNING_OPTS', "$PICFLAGS", "$USER_CFLAGS", "$USER_CFLAGS_GLOBAL",
                   "-DFAKEID=$FAKEID", "-DARCADIA_ROOT=${ARCADIA_ROOT}", "-DARCADIA_BUILD_ROOT=${ARCADIA_BUILD_ROOT}")
        blk.append("CXXFLAGS", "$CXX_WARNING_OPTS", "$CFLAGS", self.cxx_flags, "$USER_CXXFLAGS")
        blk.append("CONLYFLAGS", self.c_only_flags, "$USER_CONLYFLAGS")
        blk.emit("CXX_COMPILER_UNQUOTED", self.tc.local_path("cxx_compiler"))
        blk.emit("CXX_COMPILER", "${quo:CXX_COMPILER_UNQUOTED}")
        blk.emit("NOGCCSTACKCHECK", "yes")
        blk.emit("USE_GCCFILTER", preset("USE_GCCFILTER") or "yes")
        blk.emit("USE_GCCFILTER_COLOR", preset("USE_GCCFILTER_COLOR") or "yes")
        blk.emit("SFDL_FLAG", self.sfdl_flags, '-o', '$SFDL_TMP_OUT')
        blk.emit('WERROR_FLAG', '-Werror', '-Wno-error=deprecated-declarations')
        blk.emit('USE_ARC_PROFILE', to_yes_no(get_from_var(arc_profile_alias)))
        blk.emit("COMPILER_ENV", reformat_env(self.tc.env, values_sep=":"))
        blk.emit('DEBUG_INFO_FLAGS', '-g')

        platform = self.tc.params.get('platform', {}).get(self.target_os.os, None)
        if platform:
            blk.emit('COMPILER_PLATFORM', platform)

        if "emit_extra" in self.tc.params:
            for x, y in self.tc.params["emit_extra"]:
                blk.emit(x, y)
        blk.add("""
  when ($NO_COMPILER_WARNINGS == \"yes\") {
      CFLAGS+= -w
  }
  when ($NO_OPTIMIZE == \"yes\") {
      OPTIMIZE=-O0
  }
  when ($SAVE_TEMPS ==  \"yes\") {
      CXXFLAGS += -save-temps
  }
  when ($NOGCCSTACKCHECK != \"yes\") {
      FSTACK+= -fstack-check
  }
  when ($NO_WSHADOW == \"yes\") {
      CFLAGS += -Wno-shadow
  }

  macro MSVC_FLAGS(Flags...) {
      # TODO: FIXME
      ENABLE(UNUSED_MACRO)
  }
""")
        # specific options for LD linker
        final_ld_libs = self.gcc_sys_lib
        final_ld_libs.append("-lc")
        blk.emit("FINAL_LD_LIBS", final_ld_libs)
        # TODO: wrap cxx compiler path in ${extfile:}
        blk.emit("LINK_EXE_CMD", quote_path(self.tc.local_path("cxx_compiler")), "-o")

    def print_compiler_cmd(self, blk):
        common_args = [
            '$EXTRA_C_FLAGS',
            '-c',
            '-o',
            '${output:SRC%s.o}' % cross_suffix(),
            '${input:SRC}',
            '${pre=-I:INCLUDE}',
        ]

        kv_opts = [
            '${hide;kv:"p CC"}',
            '${hide;kv:"pc green"}',
        ]

        blk.emit('GCC_COMPILE_FLAGS', common_args)
        blk.emit('EXTRA_C_FLAGS')
        blk.emit('EXTRA_COVERAGE_OUTPUT', '${output;noauto;hide:SRC%s.gcno}' % cross_suffix())
        blk.emit('YNDEXER_OUTPUT', '${output;noauto:SRC%s.ydx.pb2}' % cross_suffix())
        if str2bool(preset('DUMP_COMPILER_DEPS')):
            emit('DUMP_DEPS', '-MD', '${output;hide;noauto:SRC.o.d}')
        elif str2bool(preset('DUMP_COMPILER_DEPS_FAST')):
            emit('DUMP_DEPS', '-E', '-M', '-MF', '${output;noauto:SRC.o.d}')

        blk.emit('EXTRA_COMPILE_OPTS', self.extra_compile_opts)
        blk.append('EXTRA_OUTPUT')

        self.cxx_args = ['$GCCFILTER', '$YNDEXER_ARGS', '$CXX_COMPILER', '$TARGET_OPT', '$GCC_COMPILE_FLAGS', '$CXXFLAGS', '$EXTRA_OUTPUT', '$EXTRA_COMPILE_OPTS', '$COMPILER_ENV'] + kv_opts
        self.c_args = ['$GCCFILTER', '$YNDEXER_ARGS', '$C_COMPILER', '$TARGET_OPT', '$GCC_COMPILE_FLAGS', '$CFLAGS', '$CONLYFLAGS', '$EXTRA_OUTPUT', '$EXTRA_COMPILE_OPTS', '$COMPILER_ENV'] + kv_opts

        self.extend_c_args()
        self.extend_cxx_args()

        blk.add("""
  macro _SRC_cpp(SRC, OPTIONS...) {
      MACRO_PROP(CMD """ + " ".join(self.cxx_args) + ")" + """
  }
  macro _SRC_c(SRC, OPTIONS...) {
      MACRO_PROP(CMD """ + " ".join(self.c_args) + ")" + """
  }
  macro _SRC_m(SRC, OPTIONS...) {
      MACRO_PROP(CMD $SRC_c($SRC $OPTIONS))
  }
  macro _SRC_masm(SRC, OPTIONS...) {
  }""")

    def extend_c_args(self):
        pass

    def extend_cxx_args(self):
        pass

    def check_system_headers(self):
        if not opts().resolve_tc_pathes:
            return

        def create_env():
            compiler_env = {}
            for key in self.tc.env:
                compiler_env[key] = ';'.join([self.tc.convert_path(x) for x in self.tc.env[key] if x])
            return compiler_env

        fake_src = os.path.join(ymake().arc_root, 'build', 'scripts', '_check_compiler.cpp')
        try:
            if get_stderr([self.tc.local_path("cxx_compiler"), '-E', fake_src], verbose=False, env=create_env()):
                raise FailedCmdException
        except FailedCmdException:
            logging.error('Seems that your system does not have system headers. Stop build. You should install them manually.')
            if l_system().os == "DARWIN":
                logging.error('Run "xcode-select --install" please.')
            elif l_system().os == "LINUX":
                logging.error('Install libc6-dev package (for Ubuntu) or similar please.')
            sys.exit(1)

    def verify_compiler(self):
        self.check_system_headers()

t_gcc_codegens = {}


def gcc_codegen(tc, target_type, stdcxx):
    key = str(tc.name) + '@' + str(target_type) + '@' + str(stdcxx)
    if key not in t_gcc_codegens:
        t_gcc_codegens[key] = GCCCodeGen(tc, target_type, stdcxx).write()
    return t_gcc_codegens[key]

t_gcc_build_type = {}


def gcc_build_type(tc, build_type):
    key = str(tc.name) + '@' + str(build_type)
    if key not in t_gcc_build_type:
        t_gcc_build_type[key] = GCCBuildType(tc, build_type).write()
    return t_gcc_build_type[key]


class GCCCodeGen(BuildTypeBlock):
    def __init__(self, tc, target_type, stdcxx):
        super(GCCCodeGen, self).__init__("name=CodeGen.tc={0}.tType={1}.stdcxx={2}".format(tc.name, target_type, stdcxx))
        # Currently not used.
        gcc_ver = version_as_list(tc.params["gcc_version"])
        self.append("C_WARNING_OPTS", "-Wno-deprecated")
        self.append("CXX_WARNING_OPTS", "-Wno-invalid-offsetof")
        self.append("CXX_WARNING_OPTS", "-Wno-attributes")

        if tc.type == "clang" and gcc_ver >= [3, 9]:
            self.append("CXX_WARNING_OPTS", "-Wno-undefined-var-template")

        if target_type == 'i386':
            self.append("CFLAGS", "-march=pentiumpro")
            self.append("CFLAGS", "-mtune=pentiumpro")
        self.append("C_DEFINES", "-D__LONG_LONG_SUPPORTED")
        if stdcxx == "11":
            self.append("CXXFLAGS", "-std=c++11")
        elif stdcxx == "14":
            self.append("CXXFLAGS", "-std=c++14")
        else:
            self.append("CXXFLAGS", "-std=gnu++03")


class GCCBuildType(BuildTypeBlock):
    def __init__(self, tc, build_type):
        super(GCCBuildType, self).__init__("name=GCCBuildType.bt=*.tc={0}".format(tc.name))

        if build_type == "valgrind" or build_type == "valgrind-release":
            self.sub_block("bt=valgrind|valgrind-release")\
                .append("C_DEFINES", "-DWITH_VALGRIND=1")

        if build_type == "debug":
            self.sub_block("bt=debug")\
                .append("CFLAGS", "$FSTACK")\
                .emit("RLGEN_FLAGS", "-T0")\
                .emit("RAGEL6_FLAGS", "-CT0")

        if build_type == "release" or build_type == "valgrind-release":
            self.sub_block("bt=release|valgrind-release")\
                .append("CFLAGS", "-DNDEBUG")\
                .append("OPTIMIZE", "-O2")\
                .append("CFLAGS", "$OPTIMIZE")

        if build_type == "relwithdebinfo":
            self.sub_block("bt=relwithdebinfo")\
                .append("CFLAGS", "-UNDEBUG")\
                .append("OPTIMIZE", "-O2")\
                .append("CFLAGS", "$OPTIMIZE")

        if build_type == "coverage":
            self.sub_block("bt=coverage")\
                .append("CFLAGS", "-fprofile-arcs", "-ftest-coverage")\
                .append("EXTRA_OUTPUT", "${output;noauto;hide:SRC%s.gcno}" % cross_suffix())

        if build_type == "profile":
            self.sub_block("bt=profile")\
                .append("OPTIMIZE", "-O2")\
                .append("CFLAGS", "$OPTIMIZE", "-DNDEBUG", "-fno-omit-frame-pointer")

        if build_type == "gprof":
            self.sub_block("bt=gprof")\
                .append("OPTIMIZE", "-O2")\
                .append("CFLAGS", "$OPTIMIZE", "-DNDEBUG", "-pg", "-fno-omit-frame-pointer")\
                .append("LDFLAGS", "-pg")


class SystemGCC(object):
    full_gcc_runtime = ["libsupc++.a", "libgcc.a", "libgcc_eh.a"]
    min_gcc_runtime = ["libgcc.a"]
    gcc_sys_libs_names = {"LINUX": full_gcc_runtime, "FREEBSD": full_gcc_runtime, "DARWIN": min_gcc_runtime, "CYGWIN": full_gcc_runtime}

    def __init__(self, tc):
        self.tc = tc
        if not os.path.exists(tc.cxx_compiler):
            logging.error("Supplied compiler " + tc.cxx_compiler + " must be available in the system."
                                                                   " (This path doesn't exist). Stop build.")
            sys.exit(1)

        try:
            gcc_version = get_stdout([tc.cxx_compiler, "-dumpversion"])
            v = version_as_list(gcc_version)
            tc.name = "gcc{0}{1}".format(v[0], v[1])
            tc.params["gcc_version"] = gcc_version
        except FailedCmdException:
            logging.error('Could not get system gcc version with command {0} -dumpversion.'
                          ' Stop build'.format(tc.cxx_compiler))
            sys.exit(1)

        self.tc.params['ar'] = 'ar'  # this is true for gcc and now for clang. it is used in LD-part

        if tc.type == "clang":
            try:
                clang_version_output = get_stdout([tc.cxx_compiler, "--version"])
                p = re.compile("^[a-zA-Z0-9 \-]+ version (?P<version>[0-9\.]+)")
                m = p.match(clang_version_output)
                if m:
                    v = version_as_list(m.group('version'))
                    tc.name = "clang{0}{1}".format(v[0], v[1])
                else:
                    logging.error('Could not extract system clang version from "clang --version" command. Output: {0}'.
                                  format(clang_version_output))
                    tc.name = 'clang'
            except FailedCmdException:
                logging.error('Could not get system clang version with command {0} --version.'.format(tc.cxx_compiler))
                tc.name = 'clang'  # seems not critical

        try:
            self.gcc_target = preset("GCC_TARGET") or get_stdout(
                [tc.cxx_compiler, "-dumpmachine"])
        except FailedCmdException:
            logging.error('Could not get system gcc target platform with command {0} -dumpmachine. Stop build.'
                          .format(tc.cxx_compiler))
            sys.exit(1)

        if self.gcc_target.find("freebsd") >= 0:
            self.target_os_id = "FREEBSD"
        elif self.gcc_target.find("linux") >= 0:
            self.target_os_id = "LINUX"
        elif self.gcc_target.find("darwin") >= 0:
            self.target_os_id = "DARWIN"
        elif self.gcc_target.find("cygwin") >= 0:
            self.target_os_id = "CYGWIN"
        else:
            logging.error(tc.cxx_compiler + ": unknown target platform {0}. Stop build!".format(self.gcc_target))
            sys.exit(1)
        target_os = t_system(self.target_os_id, l_system().type)

        if not (gcc_version and self.gcc_target):
            logging.error("GCC compiler can't be used properly. Stop build!")
            sys.exit(1)

        if target_os.x86:
            if target_os.arch == 32:
                self.arch_opt = "-m32"
            else:
                self.arch_opt = "-m64"
        else:
            self.arch_opt = '-fsigned-char'

        self.tc.params['arch_opt'] = self.arch_opt

        self.init_compiler_libs()
        self.init_sys_lib()

        self.tc.params["supported_codegen_opts"] = []

        if target_os.x86:
            self.init_sse()

        self.tc.params["have_valgrind_headers"] = check_valgrind_headers()

    def init_libs_linux(self):
        self.gcc_platform_location = "/usr/lib"

    def init_libs_freebsd(self):
        # this does not work apparently
        try:
            self.gcc_platform_location = get_stderr([self.tc.cxx_compiler, "-v"])
        except FailedCmdException:
            logging.error('Could not define gcc platform location on freebsd. Stop build.')
            sys.exit(1)

    def init_libs_win(self):
        self.gcc_platform_location = None
        logging.debug("Not implemented gcc for win.")

    def init_compiler_libs(self):
        # logging.error("tc.name = " + self.tc.name)
        self.gcc_full_lib_dir = preset("gcc_full_lib_dir")
        if not self.gcc_full_lib_dir:
            try:
                self.gcc_full_lib_dir = os.path.dirname(
                    get_stdout([self.tc.cxx_compiler, self.arch_opt, "-print-file-name=libsupc++.a"]))
            except FailedCmdException:
                gcc_compiler_libs = {"LINUX": self.init_libs_linux, "FREEBSD": self.init_libs_freebsd, "DARWIN": self.init_libs_linux,
                                     "WINDOWS": self.init_libs_win}
                gcc_compiler_libs[l_system().os]()
                self.gcc_full_lib_dir = os.path.join(self.gcc_platform_location, "gcc", self.gcc_target, self.tc.params["gcc_version"])

    def find_sys_lib(self, name):
        try:
            fullname = get_stdout([self.tc.cxx_compiler, "-print-file-name={0}".format(name)])
            if os.path.isfile(fullname):
                return os.path.normpath(fullname)
        except FailedCmdException:
            return None

    def init_sys_lib(self):
        gcc_sys_lib = []
        emit_extra = []
        for sys_lib in self.gcc_sys_libs_names[self.target_os_id]:
            resolved_name = self.find_sys_lib(sys_lib)
            if resolved_name:
                gcc_sys_lib.append(resolved_name)
            elif sys_lib == 'libsupc++.a':
                emit_extra.append(('USE_LIBCXXRT', 'yes'))
        self.tc.params["sys_lib"] = {self.target_os_id: gcc_sys_lib}
        self.tc.params["emit_extra"] = emit_extra

    def init_sse(self):
        # TODO: this stuff is only a compiler capabilities check, must be moved to CodeGen
        supported_sse_opts = []
        fake_src = os.path.join(ymake().arc_root, 'build', 'scripts', '_check_compiler.cpp')

        for opt in gcc_sse_opt_names:
            try:
                if not get_stderr([self.tc.cxx_compiler, opt, '-E', fake_src]):
                    supported_sse_opts += [opt]
            except FailedCmdException:
                pass

        self.tc.params["supported_codegen_opts"] = supported_sse_opts


class GnuGCC(GNU):
    def __init__(self, tc, target_os, build_type):
        super(GnuGCC, self).__init__(tc, target_os, build_type)

        gcc_version = version_as_list(tc.params['gcc_version'])

        if gcc_version >= version_as_list('4.9.0'):
            self.c_flags.append('-fno-delete-null-pointer-checks')
            self.c_flags.append('-fabi-version=8')


class ClangCC(GNU):
    def __init__(self, tc, target_os, build_type):
        super(ClangCC, self).__init__(tc, target_os, build_type)

        self.target_opt = filter(None, [self.tc.params.get('target_opt')])

        if target_os.os == 'DARWIN':
            self.target_opt.append('-mmacosx-version-min=10.9')
        elif target_os.os == 'IOS':
            self.target_opt.append('-mios-version-min=7.0')

        self.sfdl_flags += ['-Qunused-arguments']

    def print_compiler_settings(self, blk):
        blk.emit("CLANG", "yes")
        blk.emit("ARCH_OPT", self.tc.params.get('arch_opt', ''))
        blk.emit("TARGET_OPT", self.target_opt)
        blk.emit("CLANG_VER", self.tc.params["gcc_version"])

        v = version_as_list(self.tc.params['gcc_version'])

        if v >= version_as_list('3.6'):
            self.c_flags.extend([
                '-Wno-inconsistent-missing-override'
            ])

        self.print_gnu_compiler_common_settings(blk)


class ICC(GNU):
    # example project: cv/imgclassifiers/danet/nn_runner
    gcc_fstack = ['-fno-stack-protector']

    def __init__(self, tc, target_os, build_type):
        super(ICC, self).__init__(tc, target_os, build_type)

        if target_os.arch == 32:
            logging.error("Your system is i386 but we have icc toolchain only for 64-bit.")
            sys.exit()
        self.extend_c_flags()

    @staticmethod
    def get_gcc_flags(tc):
        gcc_dir = os.path.join(os.path.dirname(os.path.dirname(tc.local_path('c_compiler'))), 'gcc')
        return ['-gcc-name=gcc-4.8', '-cxxlib={0}'.format(quote_path(gcc_dir)), '-isystem/usr/include/x86_64-linux-gnu']

    def extend_c_flags(self):
        # TODO: wrap gcc_dir in ${extfile:}
        self.c_flags.extend(ICC.get_gcc_flags(self.tc))

    def print_compiler_settings(self, blk):
        blk.emit("ICC", "yes")
        blk.emit("ICC_VER", self.tc.params["gcc_version"])
        self.print_gnu_compiler_common_settings(blk)


class ICCwithMIC(ICC):
    def __init__(self, tc, target_os, build_type):
        super(ICCwithMIC, self).__init__(tc, target_os, build_type)
        self.mpss_sysroots = self.tc.params['mpss_sysroots']

    def extend_c_flags(self):
        pass

    def extend_c_args(self):
        self.c_args.append('${{env:"MPSS_SYSROOTS={}"}}'.format(self.mpss_sysroots))

    def extend_cxx_args(self):
        self.cxx_args.append('${{env:"MPSS_SYSROOTS={}"}}'.format(self.mpss_sysroots))


class LD(object):
    def __init__(self, tc, target_os, build_type):
        self.ld_thread_lib = self.ld_flags = self.libresolv = self.thirdparty_libs = None
        self.tc = tc
        self.build_type = build_type

        force_libtool = target_os.os == 'DARWIN' and 'libtool' not in str(self.tc.params.get('ar'))  # XXX

        if force_libtool:  # we use system libtool (ar), as and ld on host mac os x
            self.ar = 'libtool'
        else:
            if l_system().os == 'DARWIN':
                default_ar = 'libtool'
            else:
                default_ar = '{}/gcc/bin/ar'.format(tc.name_marker) if not opts().resolve_tc_pathes else 'ar'
            self.ar = preset("AR") or self.tc.local_path('ar') or default_ar

        self.ar_plugin = self.tc.params.get('ar_plugin', 'None')

        os_opts = {"LINUX": LD.init_linux, "FREEBSD": LD.init_freebsd, "DARWIN": LD.init_darwin,
                   "WINDOWS": LD.init_win, "ANDROID": LD.init_android, "IOS": LD.init_ios, "CYGWIN": LD.init_cygwin}

        os_opts[target_os.os](self)
        self.dwarf_tool = tc.params.get('dwarf_tool', {}).get(target_os.os, '')

        self.os = target_os.os

        self.sdk_ldflags = []
        if target_os.os == 'DARWIN':
            self.sdk_ldflags += ['-macosx_version_min', '10.9']
        elif target_os.os == 'IOS':
            self.sdk_ldflags += ['-ios_version_min', '7.0']

    def init_cygwin(self):
        self.init_linux()
        self.rdynamic = ''
        self.use_stdlib = ''

    def init_win(self):
        logging.debug("ld usage for Windows is not implemented!")
        self.ld_thread_lib = self.ld_flags = self.rdynamic = self.start_group = \
            self.end_group = self.ld_flags = self.soname = self.use_stdlib = \
            self.ld_stripflag = self.pie = self.ar_type = self.dwarf_info = ''

    def init_linux(self):
        self.ld_thread_lib = "-lpthread"
        self.ld_flags = ["-ldl", "-lrt"]
        self.rdynamic = '-rdynamic'
        self.start_group = "-Wl,--start-group"
        self.end_group = "-Wl,--end-group"
        ld_preset = preset("LDFLAGS")
        if ld_preset:
            self.ld_flags.append(ld_preset)
        self.ld_flags.append("-Wl,--no-as-needed")
        self.pie = ''
        self.soname = '-soname'
        self.use_stdlib = '-nodefaultlibs'
        self.ld_stripflag = '-s'
        if not opts().resolve_tc_pathes:
            self.libresolv = "-lresolv"
        self.ar_type = 'AR'
        self.dwarf_info = ''

    def init_android(self):
        self.ld_thread_lib = ""
        self.ld_flags = ["-ldl", "-lsupc++"]
        self.rdynamic = '-rdynamic'
        self.start_group = "-Wl,--start-group"
        self.end_group = "-Wl,--end-group"
        ld_preset = preset("LDFLAGS")
        if ld_preset:
            self.ld_flags.append(ld_preset)
        self.ld_flags.append("-Wl,--no-as-needed")
        self.pie = '-pie'
        self.soname = '-soname'
        self.use_stdlib = '-nodefaultlibs'
        self.ld_stripflag = '-s'
        if not opts().resolve_tc_pathes:
            self.libresolv = "-lresolv"
        self.ar_type = 'AR'
        self.dwarf_info = ''

    def init_darwin(self):
        self.ld_thread_lib = '-lpthread'
        self.libresolv = '-lresolv'
        self.rdynamic = ''
        self.start_group = ''
        self.end_group = ''
        self.pie = ''
        self.ld_flags = []

        if self.tc.type != 'clang':
            self.ld_flags.append('-Wl,-no_compact_unwind')

        self.soname = '-install_name'
        self.use_stdlib = '-nodefaultlibs'
        self.ld_stripflag = ''
        self.ar_type = 'LIBTOOL'
        self.dwarf_info = '&& $DWARF_TOOL $TARGET -o ${output;pre=$REALPRJNAME.dSYM/Contents/Resources/DWARF/:REALPRJNAME}' if not preset('NO_DEBUGINFO') else ''

    def init_ios(self):
        self.init_darwin()
        self.ld_thread_lib = ''

    def init_freebsd(self):
        self.ld_thread_lib = "-lthr"
        self.rdynamic = '-rdynamic'
        self.start_group = "-Wl,--start-group"
        self.end_group = "-Wl,--end-group"
        self.soname = '-soname'
        self.pie = ''
        self.use_stdlib = '-nodefaultlibs'
        self.ld_stripflag = '-s'
        self.ar_type = 'AR'
        self.dwarf_info = ''

    def print_linker_settings(self, blk):
        blk.emit("LIBRT", "-lrt")
        blk.emit("MD5LIB", "-lcrypt")
        blk.emit("MPROFLIB", "")
        blk.emit("THREADLIB", self.ld_thread_lib)
        blk.emit("LIBRESOLV", self.libresolv)
        blk.emit("PROFFLAG", "-pg")
        blk.emit("OBJADDE", "")
        blk.emit("THIRDPARTY_OBJADD", "")
        blk.emit('THIRDPARTY_LIBS', self.thirdparty_libs or "")
        blk.append("LDFLAGS", self.ld_flags)
        blk.append("LDFLAGS_GLOBAL", "")
        blk.emit("LIBS", "")
        blk.emit("LINK_OPTIONS_START", "${rootrel:SRCS_GLOBAL}", '$EXPORTS_VALUE', self.start_group, self.rdynamic, "$THREADLIB", "$THIRDPARTY_LIBS")
        blk.emit('LINK_DYN_OPTIONS_START', '${rootrel:SRCS_GLOBAL}', '$EXPORTS_VALUE', self.start_group, "$THIRDPARTY_LIBS", '$OBJADDE', '$OBJADDE_LIB')
        blk.emit("LINK_OPTIONS_END", self.end_group, "$THREADLIB", "$LDFLAGS", "$LDFLAGS_GLOBAL", "$OBJADDE",
                 "$OBJADDE_LIB", "$THIRDPARTY_OBJADD", "$MPROFLIB", "$FINAL_LD_LIBS")
        blk.emit("AR_TOOL", self.ar)
        blk.emit('AR_TYPE', self.ar_type)
        blk.emit("LD_STRIP_FLAG", self.ld_stripflag)
        blk.emit("STRIP_FLAG")
        blk.emit("EXPORTS_VALUE")
        blk.emit("DWARF_TOOL", self.dwarf_tool)
        # NOTE: compiler defines LINK_EXE_CMD
        blk.emit("LINK_EXE_FLAGS1", "$LINK_OPTS")
        blk.emit("LINK_OPTS", "$USE_STDLIB")
        blk.emit("USE_STDLIB", '' if preset('SANITIZER_TYPE') else self.use_stdlib)
        blk.emit("LINKER_ENV", reformat_env(self.tc.env, values_sep=":"))
        blk.add("""
  when ($EXPORTS_FILE) {
      EXPORTS_VALUE=-Wl,--version-script=${input:EXPORTS_FILE}
  }
""")
        if self.build_type == "coverage":
            sub_blk = blk.sub_block("bt=coverage")
            sub_blk.emit("USE_STDLIB", '')
            sub_blk.append("LDFLAGS", "-fprofile-arcs", "-ftest-coverage")

    def print_linker_cmd(self, blk):
        add = self.get_linker_cmd_args()
        sdk_flags = ['-Wl,{}'.format(opt) for opt in self.sdk_ldflags]

        blk.emit("LINK_LIB_IMPL", '${input:"build/scripts/link_lib.py"}', '${quo:AR_TOOL}', '$AR_TYPE', '$ARCADIA_BUILD_ROOT', self.ar_plugin)
        blk.emit('LINK_LIB', '$YMAKE_PYTHON', '$LINK_LIB_IMPL', '$TARGET', '$AUTO_INPUT', '${kv;hide:"p AR"}',
                 '${kv;hide:"pc light-red"}', '${kv;hide:"show_out"}', '$LINKER_ENV', *add)

        blk.emit("LINK_EXE", '${cwd:ARCADIA_BUILD_ROOT}', '$GCCFILTER', '$LINK_EXE_CMD', '$TARGET', '$TARGET_OPT', self.tc.params.get('sdk_lib', None),
                 '$LINK_EXE_FLAGS1', '$AUTO_INPUT', '$LINK_OPTIONS_START', '${rootrel:PEERS}', '$LINK_OPTIONS_END', self.pie,
                 '$STRIP_FLAG', sdk_flags, self.dwarf_info, '${kv;hide:"p LD"}', '${kv;hide:"pc light-blue"}', '${kv;hide:"show_out"}', '$LINKER_ENV', *add)

        blk.emit("LINK_DYN_LIB",
                 '${cwd:ARCADIA_BUILD_ROOT}', '$YMAKE_PYTHON', '${input:"build/scripts/link_dyn_lib.py"}',
                 '--arch', self.os, '--target', '$TARGET', '$LINK_ADD',
                 '$LINK_EXE_CMD', '$TARGET', '$TARGET_OPT', self.tc.params.get('sdk_lib', None), '-shared', '-Wl,' + self.soname + ',$SONAME',
                 '$LINK_DYN_OPTIONS_START', '$AUTO_INPUT', '${rootrel:PEERS}', '$LINK_OPTS', sdk_flags, '$LINK_OPTIONS_END', '$STRIP_FLAG',
                 self.dwarf_info, '${kv;hide:"p LD"}',
                 '${kv;hide:"pc light-blue"}', '${kv;hide:"show_out"}', '$LINKER_ENV', *add)

        blk.emit("LINK_FAT_OBJECT",
                 '$YMAKE_PYTHON', '${input:"build/scripts/link_fat_obj.py"}', '--linker',
                 '${quo:LINK_EXE_CMD}', '--python', '${quo:YMAKE_PYTHON}', '--ar', '$LINK_LIB_IMPL', '--obj', '$TARGET', '--lib', '${output:REALPRJNAME.a}',
                 '--target-opt', '${quo:TARGET_OPT}', '--arch', self.os, '--auto-input', '$AUTO_INPUT', '--global-srcs', '$SRCS_GLOBAL', '--peers', '$PEERS',
                 ['--linker-opt={}'.format(opt) for opt in self.sdk_ldflags],
                 '${kv;hide:"p LD"}', '${kv;hide:"pc light-blue"}', '${kv;hide:"show_out"}', '$LINKER_ENV')

    def get_linker_cmd_args(self):
        return []


class ICCLD(LD):
    def __init__(self, tc, target_os, build_type):
        super(ICCLD, self).__init__(tc, target_os, build_type)
        self.ar = tc.local_path('ar')
        if self.ar is None:
            self.ar = preset('AR') or 'ar'
        self.libresolv = "-lresolv"
        self.extend_ld_flags(tc)

    def get_linker_cmd_args(self):
        return ['${{env:"PATH={}/gcc/bin"}}'.format(self.tc.name_marker)]

    def extend_ld_flags(self, tc):
        self.ld_flags.extend(ICC.get_gcc_flags(tc))


class ICCwithMICLD(ICCLD):
    def __init__(self, tc, target_os, build_type):
        super(ICCwithMICLD, self).__init__(tc, target_os, build_type)
        self.mpss_sysroots = self.tc.params['mpss_sysroots']

    def get_linker_cmd_args(self):
        return ['${{env:"MPSS_SYSROOTS={}"}}'.format(self.mpss_sysroots)]

    def extend_ld_flags(self, tc):
        pass

def detect_msvc_cxx():
    try:
        # https://developercommunity.visualstudio.com/content/problem/2813/cant-find-registry-entries-for-visual-studio-2017.html
        import _winreg
        vskey = _winreg.OpenKey(_winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\Wow6432Node\Microsoft\VisualStudio\SxS\VS7")
        vid = 0
        vs_max_version = ""
        vs_base_path = ""
        while True:
            try:
                name, value, type = _winreg.EnumValue(vskey, vid)
                vid += 1
                if name > vs_max_version and os.path.exists(value):
                    vs_max_version = name
                    vs_base_path = value
            except _winreg.error:
                break
        vs_tools_version_file = os.path.join(vs_base_path, r"VC\Auxiliary\Build\Microsoft.VCToolsVersion.default.txt")
        if not os.path.exists(vs_tools_version_file):
            raise Exception("Can not find Microsoft.VCToolsVersion.default.txt")
        with open(vs_tools_version_file) as f:
            vs_tools_version = f.read().strip()
        vs_tools_base_path = os.path.join(vs_base_path, r"VC\Tools\MSVC", vs_tools_version)
        cxx_compiler = os.path.join(vs_tools_base_path, r"bin\HostX64\x64\cl.exe")  # detect arch?
        if not os.path.exists(cxx_compiler):
            raise Exception("Can not find cxx compiler for version " + vs_tools_version)
        return cxx_compiler
    except Exception, e:
        logging.error("Could not detect cxx compiler, error is " + str(e))
        sys.exit(1)


def detect_msvc(tc):

    # TODO: better condition for detect_msvc_cxx() run
    if tc.cxx_compiler == "cl.exe":
        tc.cxx_compiler = detect_msvc_cxx()

    cl_full_path = os.path.normcase(tc.cxx_compiler)
    if cl_full_path is None:
        logging.error("Could not find " + tc.cxx_compiler + ", make sure it is in you PATH")

    # we can enter this function only when cl.exe is an executable, i.e. config platform is Windows
    try:
        import _winreg
        # TODO: get more robust code from contrib/tools/python/src/Lib/distutils/msvc9compiler.py ?
        vskey = _winreg.OpenKey(_winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\Wow6432Node\Microsoft\VisualStudio\SxS\VS7")
        vid = 0
        vsver = None
        while True:
            try:
                name, value, type = _winreg.EnumValue(vskey, vid)
            except _winreg.error:
                break
            vid += 1
            value = os.path.normcase(value)
            if os.path.commonprefix([value, cl_full_path]) == value:
                vsver = name
                logging.error("In toolchain " + tc.name + ", detected system cl compiler from VS version " + name)
                if name == "15.0":
                    tc.name = "msvc2015"
                    sdk_kit = "KitsRoot10"
                    vc_shared_version = "14.0"
                else:
                    logging.error("Do not know how to use VS version " + name)
                    sys.exit(1)
                tc.toolchain_root = value
                break
    except ImportError:
        vsver = None

    if vsver is None:
        logging.error("Could not find system msvc configuration for " + tc.cxx_compiler)
        sys.exit(1)

    sdk_root = None
    vid = 0
    sdk_key = _winreg.OpenKey(_winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\Wow6432Node\Microsoft\Windows Kits\Installed Roots")
    while True:
        try:
            name, value, type = _winreg.EnumValue(sdk_key, vid)
        except _winreg.error:
            break
        vid += 1
        if name == sdk_kit:
            sdk_root = value
            break
    if sdk_root is None:
        logging.error("Could not find installed Platform SDK " + sdk_kit + " for " + tc.cxx_compiler)
        sys.exit(1)

    sdk_version = ""
    vid = 0
    while True:
        try:
            sdk_version = max(_winreg.EnumKey(sdk_key, vid), sdk_version)
            vid += 1
        except _winreg.error:
            break
    if not sdk_version:
        logging.error("Could not find any versions of installed Platform SDK " + sdk_kit + " for " + tc.cxx_compiler)
        sys.exit(1)

    try:
        vc_key = _winreg.OpenKey(_winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\Wow6432Node\Microsoft\VisualStudio\SxS\Vc7")
    except Exception:
        logging.error("Could not find VC shared version " + vc_shared_version + " for " + tc.cxx_compiler)
        sys.exit(1)

    vc_shared = ""
    vid = 0
    while True:
        try:
            name, value, type = _winreg.EnumValue(vc_key, vid)
            vid += 1
        except _winreg.error:
            break
        if name == vc_shared_version:
            vc_shared = value
    if not vc_shared:
        logging.error("Could not find VC shared version " + vc_shared_version + " for " + tc.cxx_compiler)
        sys.exit(1)

    # TODO: get all of these from registry, in a different place
    tc.type = "msvc"
    tc.params["sdk_root"] =  sdk_root
    tc.params["sdk_version"] = sdk_version
    tc.params["vc_shared"] = vc_shared
    tc.params["system_msvc"] = True

    prog_dir = os.path.dirname(tc.cxx_compiler)
    tc.params["cxx_compiler"] = tc.cxx_compiler
    tc.params["c_compiler"] = tc.cxx_compiler
    tc.params["lib"] = os.path.join(prog_dir, "lib.exe")
    tc.params["link"] = os.path.join(prog_dir, "link.exe")
    tc.params["masm_compiler"] = os.path.join(prog_dir, "ml64.exe")
    tc.params["env"] = {"PATH": [prog_dir, os.path.dirname(prog_dir)]}
    tc.name_marker = None


class WIN32_WINNT(object):
    Macro = '_WIN32_WINNT'
    Windows7 = '0x0601'
    Windows8 = '0x0602'


class MSVCBase(object):
    def __init__(self, tc):
        if 'sdk_root' not in tc.params:
            raise Exception('There should be "sdk_root" parameter in MSVC toolchain')

        self._tc = tc
        self._tc_sdk_root = tc.local_path('sdk_root')
        if self._tc_sdk_root.endswith('/'):
            self._tc_sdk_root = self._tc_sdk_root[:-1]

        self.ide_msvs = tc.params.get('ide_msvs', '')
        self.under_wine = str2bool(tc.params.get('wine'))
        self.fix_msvc_output = not str2bool(tc.params.get('disable_fix_msvc_output', ''))
        self.sdk_root = '$(VSInstallDir)' if self.ide_msvs else self._tc_sdk_root

    def sdk_path(self, v):
        path = self._tc.local_path(v)
        if self.ide_msvs:
            path = path.replace(self._tc_sdk_root, self.sdk_root)
        return path


class MSVC(MSVCBase):
    def __init__(self, build_settings, tc):
        super(MSVC, self).__init__(tc)
        self.build_settings = build_settings
        self.parent_blocks = []  # TODO: get rid of parent_blocks

    def print_compiler_settings(self, blk):
        target = self.build_settings.target

        compiler_c = self.sdk_path('c_compiler')
        compiler_cxx = self.sdk_path('cxx_compiler')
        compiler_masm = self.sdk_path('masm_compiler')

        win32_winnt = WIN32_WINNT.Windows7

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
        ]

        defines = [
            'WIN32',
            '_WIN32',
            '_WINDOWS',
            '_CRT_SECURE_NO_WARNINGS',
            '_CRT_NONSTDC_NO_WARNINGS',
            '_USE_MATH_DEFINES',
            '__STDC_CONSTANT_MACROS',
            '__STDC_FORMAT_MACROS',
            '_USING_V110_SDK71_',
            # TODO: check NOSSE flags
            'SSE_ENABLED=1',
            'SSE2_ENABLED=1',
            'SSE3_ENABLED=1'
        ]

        winapi_unicode = False

        defines_debug = ['_DEBUG']
        defines_release = ['NDEBUG']

        print """\
MSVC_INLINE_OPTIMIZED=yes
when ($MSVC_INLINE_OPTIMIZED == "yes") {
    MSVC_INLINE_FLAG=/Zc:inline
}
when ($MSVC_INLINE_OPTIMIZED == "no") {
    MSVC_INLINE_FLAG=/Zc:inline-
}
"""

        flags = ['/nologo', '/Zm500', '/GR', '/bigobj', '/FC', '/EHsc', '/errorReport:prompt', '$MSVC_INLINE_FLAG', '/DFAKEID=$FAKEID']
        flags += ['/we{}'.format(code) for code in warns_as_error]
        flags += ['/w1{}'.format(code) for code in warns_enabled]
        flags += ['/wd{}'.format(code) for code in warns_disabled]
        flags += target.arch_opt

        flags_debug = ['/Ob0', '/Od'] + self._gen_defines(defines_debug)
        flags_release = ['/Ox', '/Ob2', '/Oi'] + self._gen_defines(defines_release)

        flags_cxx = []

        if target.is_arm:
            masm_io = '-o ${output:SRC.obj} ${input;msvs_source:SRC}'
        else:
            masm_io = '/nologo /c /Fo${output:SRC.obj} ${input;msvs_source:SRC}'

        if is_positive('USE_UWP'):
            flags_cxx += ['/ZW', '/AI{root}/VC/lib/store/references'.format(root=self.sdk_root)]
            if not self.ide_msvs:
                flags.append('/I{root}/include/winrt'.format(root=self.sdk_root))
            win32_winnt = WIN32_WINNT.Windows8
            defines.append('WINAPI_FAMILY=WINAPI_FAMILY_APP')
            winapi_unicode = True

        emit('WIN32_WINNT', '{value}'.format(value=win32_winnt))
        defines.append('{name}=$WIN32_WINNT'.format(name=WIN32_WINNT.Macro))

        if winapi_unicode:
            defines += ['UNICODE', '_UNICODE']
        else:
            defines += ['_MBCS']

        # https://msdn.microsoft.com/en-us/library/abx4dbyh.aspx
        if is_positive('DLL_RUNTIME'):  # XXX
            flags_debug += ['/MDd']
            flags_release += ['/MD']
        else:
            flags_debug += ['/MTd']
            flags_release += ['/MT']

        if not self.ide_msvs:

            if self._tc.params.get('system_msvc'):
                for incdir in ('{root}/shared', '{root}/ucrt', '{root}/um', '{root}/winrt'):
                    option = '/I"{incdir}"'.format(incdir=incdir.format(
                        root=os.path.join(self.sdk_root, 'Include', self._tc.params.get('sdk_version'))
                    ))
                    flags.append(option)
                flags.append('/I"{vc_shared}/include"'.format(vc_shared=self._tc.params.get('vc_shared')))
            else:
                for incdir in ('{root}/include/shared', '{root}/include/ucrt', '{root}/include/um', '{root}/VC/include'):
                    option = '/I{incdir}'.format(incdir=incdir.format(root=self.sdk_root))
                    flags.append(option)

        if self.ide_msvs:
            flags += ['/FD', '/MP']
            debug_info_flags = '/Zi /FS'
        else:
            debug_info_flags = '/Z7'

        defines = self._gen_defines(defines)
        compiler_env = reformat_env(target.environ, values_sep=';')

        flags_werror = ['/WX']
        flags_sfdl = ['/E', '/C', '/P', '/Fi$SFDL_TMP_OUT']
        flags_no_optimize = ['/Od']
        flags_no_shadow = ['/wd4456', '/wd4457']
        flags_no_compiler_warnings = ['/w']

        emit('MSVC', 'yes')

        emit('CXX_COMPILER', compiler_cxx)
        emit('C_COMPILER', compiler_c)
        emit('MASM_COMPILER', compiler_masm)
        append('C_DEFINES', defines)
        emit('CFLAGS_DEBUG', flags_debug)
        emit('CFLAGS_RELEASE', flags_release)
        emit('MASMFLAGS', '')
        emit('COMPILER_ENV', compiler_env)
        emit('DEBUG_INFO_FLAGS', debug_info_flags)

        if self.build_settings.is_release:
            emit('CFLAGS_PER_TYPE', '$CFLAGS_RELEASE')
        if self.build_settings.is_debug:
            emit('CFLAGS_PER_TYPE', '$CFLAGS_DEBUG')
        if self.build_settings.is_ide:
            emit('CFLAGS_PER_TYPE', '@[debug|$CFLAGS_DEBUG]@[release|$CFLAGS_RELEASE]')

        append('CFLAGS', flags, '$CFLAGS_PER_TYPE', '$DEBUG_INFO_FLAGS', '$C_DEFINES', '$USER_CFLAGS', '$USER_CFLAGS_GLOBAL')
        append('CXXFLAGS', '$CFLAGS', flags_cxx, '$USER_CXXFLAGS')

        print '''\
when ($NO_OPTIMIZE == "yes") {{
    OPTIMIZE = {no_opt}
}}
when ($NO_COMPILER_WARNINGS == "yes") {{
    CFLAGS += {no_warn}
}}
when ($NO_WSHADOW == "yes") {{
    CFLAGS += {no_shadow}
}}
'''.format(no_opt=' '.join(flags_no_optimize), no_warn=' '.join(flags_no_compiler_warnings), no_shadow=' '.join(flags_no_shadow))

        emit("SFDL_FLAG", flags_sfdl)
        emit('WERROR_FLAG', flags_werror)
        emit("WERROR_MODE", self._tc.werror_mode or "compiler_specific")

        if self.fix_msvc_output:
            emit('CL_WRAPPER', '${YMAKE_PYTHON}', '${input:"build/scripts/fix_msvc_output.py"}', 'cl')
            emit('ML_WRAPPER', '${YMAKE_PYTHON}', '${input:"build/scripts/fix_msvc_output.py"}', 'ml')
        else:
            emit('CL_WRAPPER')
            emit('ML_WRAPPER')

        print """\
macro MSVC_FLAGS(Flags...) {
    CFLAGS($Flags)
}

macro _SRC_cpp(SRC, OPTIONS...) {
    MACRO_PROP(CMD ${cwd:ARCADIA_BUILD_ROOT} ${COMPILER_ENV} ${CL_WRAPPER} ${CXX_COMPILER} /c /Fo${output:SRC.obj} ${input;msvs_source:SRC} ${pre=/I :INCLUDE} ${CXXFLAGS} ${hide;kv:"soe"} ${hide;kv:"p CC"} ${hide;kv:"pc yellow"})
}

macro _SRC_c(SRC, OPTIONS...) {
    MACRO_PROP(CMD ${cwd:ARCADIA_BUILD_ROOT} ${COMPILER_ENV} ${CL_WRAPPER} ${C_COMPILER} /c /Fo${output:SRC.obj} ${input;msvs_source:SRC} ${pre=/I :INCLUDE} ${CFLAGS} ${hide;kv:"soe"} ${hide;kv:"p CC"} ${hide;kv:"pc yellow"})
}

macro _SRC_m(SRC, OPTIONS...) {
}

macro _SRC_masm(SRC, OPTIONS...) {
    MACRO_PROP(CMD ${cwd:ARCADIA_BUILD_ROOT} ${COMPILER_ENV} ${ML_WRAPPER} ${MASM_COMPILER} ${MASMFLAGS} """ + masm_io + """ ${kv;hide:"p AS"} ${kv;hide:"pc yellow"})
}
"""

    @staticmethod
    def print_compiler_cmd(blk):
        pass

    @staticmethod
    def _gen_defines(defines):
        return ['/D{}'.format(s) for s in defines]


class MSVCLinker(MSVCBase):
    def __init__(self, build_settings, tc):
        super(MSVCLinker, self).__init__(tc)
        self.build_settings = build_settings

    def print_linker_settings(self, blk):
        target = self.build_settings.target  # type: Platform

        linker = self.sdk_path('link')
        linker_lib = self.sdk_path('lib')

        machine = sdklibarch = vclibarch = ''

        if target.is_intel:
            if target.is_32_bit:
                machine = 'X86'
                vclibarch = None
                sdklibarch = 'x86'
            elif target.is_64_bit:
                machine = 'X64'
                vclibarch = 'amd64'
                sdklibarch = 'x64'
        elif target.is_arm and target.is_32_bit:
            machine = 'ARM'
            vclibarch = 'arm'
            sdklibarch = 'arm'

        if not machine:
            raise Exception('Unknown target platform {}'.format(str(target)))

        libpaths = []
        if self._tc.params.get("system_msvc"):
            for lib in  ('um', 'ucrt'):
                libpaths.append(
                    os.path.join(self.sdk_root, "Lib", self._tc.params.get("sdk_version"), lib, sdklibarch)
                )
            libpaths.append(
                    os.path.join(self._tc.params.get("vc_shared"), "lib", vclibarch)
            )
        else:
            for lib in ('um', 'ucrt'):
                libpaths.append(r'{root}\lib\{lib}\{arch}'.format(root=self.sdk_root, lib=lib, arch=sdklibarch))
            if is_positive('USE_UWP'):
                libpaths.append(r'{root}\VC\libs\store\references'.format(root=self.sdk_root))
            vclibs = r'{root}\VC\lib'.format(root=self.sdk_root)
            if vclibarch:
                vclibs = r'{libs}\{arch}'.format(libs=vclibs, arch=vclibarch)
            libpaths.append(vclibs)

        ignored_errors = [
            4221
        ]

        flag_machine = '/MACHINE:{}'.format(machine)

        flags_ignore = ['/IGNORE:{}'.format(code) for code in ignored_errors]

        flags_common = ['/NOLOGO', '/ERRORREPORT:PROMPT', '/SUBSYSTEM:CONSOLE', '/TLBID:1', '$MSVC_DYNAMICBASE', '/NXCOMPAT']
        flags_common += flags_ignore
        flags_common += [flag_machine]

        flags_debug_only = []
        flags_release_only = []

        if self.ide_msvs:
            flags_common += ['/INCREMENTAL']
        else:
            flags_common += ['/INCREMENTAL:NO']

        # TODO: DEVTOOLS-1868 remove restriction
        if not self.under_wine:
            if self.ide_msvs:
                flags_debug_only.append('/DEBUG:FASTLINK')
                flags_release_only.append('/DEBUG')
            else:
                # No FASTLINK for ya make, because resulting PDB would require .obj files (build_root's) to persist
                flags_common.append('/DEBUG')

        if not self.ide_msvs:
            flags_common += ['/LIBPATH:"{}"'.format(path) for path in libpaths]

        linker_env = reformat_env(target.environ, values_sep=';')
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
        emit('LINKER_ENV', linker_env)
        emit('LDFLAGS_GLOBAL', '')
        emit('LDFLAGS', '')
        emit('OBJADDE', '')

        if self.build_settings.is_release:
            emit('LINK_EXE_FLAGS_PER_TYPE', '$LINK_EXE_FLAGS_RELEASE')
        if self.build_settings.is_debug:
            emit('LINK_EXE_FLAGS_PER_TYPE', '$LINK_EXE_FLAGS_DEBUG')
        if self.build_settings.is_ide and self.ide_msvs:
            emit('LINK_EXE_FLAGS_PER_TYPE', '@[debug|$LINK_EXE_FLAGS_DEBUG]@[release|$LINK_EXE_FLAGS_RELEASE]')

        blk.emit('LINK_EXE_FLAGS', '$LINK_EXE_FLAGS_PER_TYPE')

        # TODO: DEVTOOLS-1868 remove restriction
        if self.under_wine:
            emit('LINK_EXTRA_OUTPUT')
        else:
            emit('LINK_EXTRA_OUTPUT', '/PDB:${output;noext;rootrel:REALPRJNAME.pdb}')

        if self.fix_msvc_output:
            emit('LIB_WRAPPER', '${YMAKE_PYTHON}', '${input:"build/scripts/fix_msvc_output.py"}', 'lib')
            emit('LINK_WRAPPER', '${YMAKE_PYTHON}', '${input:"build/scripts/fix_msvc_output.py"}', 'link')
        else:
            emit('LIB_WRAPPER')
            emit('LINK_WRAPPER')

        emit('LINK_WRAPPER_DYNLIB', '${YMAKE_PYTHON}', '${input:"build/scripts/link_dyn_lib.py"}', '--arch', 'WINDOWS', '--target', '$TARGET')
        emit('EXPORTS_VALUE')

        print """\
when ($EXPORTS_FILE) {
    EXPORTS_VALUE=/DEF:${input:EXPORTS_FILE}
}

LINK_LIB=${LINKER_ENV} ${cwd:ARCADIA_BUILD_ROOT} ${LIB_WRAPPER} ${LINK_LIB_CMD} /OUT:${qe;rootrel:TARGET} \
${qe;rootrel:AUTO_INPUT} $LINK_LIB_FLAGS ${hide;kv:"soe"} ${hide;kv:"p AR"} ${hide;kv:"pc light-red"}

LINK_EXE=${LINKER_ENV} ${cwd:ARCADIA_BUILD_ROOT} ${LINK_WRAPPER} ${LINK_EXE_CMD} /OUT:${qe;rootrel:TARGET} \
${LINK_EXTRA_OUTPUT} ${qe;rootrel:SRCS_GLOBAL} ${qe;rootrel:AUTO_INPUT} $LINK_EXE_FLAGS $LINK_STDLIBS $LDFLAGS $LDFLAGS_GLOBAL $OBJADDE \
${qe;rootrel:PEERS} ${hide;kv:"soe"} ${hide;kv:"p LD"} ${hide;kv:"pc blue"}

LINK_DYN_LIB=${LINKER_ENV} ${cwd:ARCADIA_BUILD_ROOT} ${LINK_WRAPPER} ${LINK_WRAPPER_DYNLIB} ${LINK_EXE_CMD} \
/DLL /OUT:${qe;rootrel:TARGET} ${LINK_EXTRA_OUTPUT} ${EXPORTS_VALUE} \
${qe;rootrel:SRCS_GLOBAL} ${qe;rootrel:AUTO_INPUT} ${qe;rootrel:PEERS} \
$LINK_EXE_FLAGS $LINK_STDLIBS $LDFLAGS $LDFLAGS_GLOBAL $OBJADDE ${hide;kv:"soe"} ${hide;kv:"p LD"} ${hide;kv:"pc blue"}

LINK_FAT_OBJECT=$YMAKE_PYTHON ${input:"build/scripts/touch.py"} $TARGET ${kv;hide:"p LD"} ${kv;hide:"pc light-blue"} ${kv;hide:"show_out"}
"""

    @staticmethod
    def print_linker_cmd(blk):
        pass


# find some system headers, utilities etc #


def check_valgrind_headers():
    valgrind_headers = ['valgrind.h', 'memcheck.h']
    possible_locations = ['/usr/include/valgrind', '/usr/local/include/valgrind']
    for location in possible_locations:
        if all(os.path.exists(os.path.join(location, x)) for x in valgrind_headers):
            return True
    return False


def find_nix_python():
    if not is_negative("USE_ARCADIA_PYTHON"):
        logging.debug("Using Arcadia python ($USE_ARCADIA_PYTHON=yes).")
        return None

    logging.debug("Using system python ($USE_ARCADIA_PYTHON=no)")
    py_config = preset('PYTHON_CONFIG') or which('python-config')
    python_dict = {}
    if not py_config:
        python_dict["PYTHON_FLAGS"] = preset("PYTHON_FLAGS") or "python-system-flags-not-found"
        python_dict["PYTHON_LDFLAGS"] = preset("PYTHON_LDFLAGS") or "python-system-ldflags-not-found"
        python_dict["PYTHON_LIBRARIES"] = preset("PYTHON_LIBRARIES") or "python-system-libs-not-found"
        python_dict["PYTHON_INCLUDE"] = preset("PYTHON_INCLUDE") or "python-system-includes-not-found"
    else:
        try:
            python_ld_flags = get_stdout([py_config, "--ldflags"])
            python_dict["PYTHON_FLAGS"] = preset("PYTHON_FLAGS") or get_stdout([py_config, "--cflags"])
            python_dict["PYTHON_LDFLAGS"] = preset("PYTHON_LDFLAGS") or python_ld_flags
            python_dict["PYTHON_INCLUDE"] = preset("PYTHON_INCLUDE") or get_stdout([py_config, "--includes"])
        except FailedCmdException:
            python_dict["PYTHON_FLAGS"] = "python-system-flags-not-found"
            python_dict["PYTHON_LDFLAGS"] = "python-system-ldflags-not-found"
            python_dict["PYTHON_INCLUDE"] = "python-system-includes-not-found"
            python_dict["PYTHON_LIBRARIES"] = "python-system-libs-not-found"
            return python_dict

        python_libs = preset("PYTHON_LIBRARIES")
        if not python_libs and python_ld_flags:
            p = re.compile("^-L([^\s]+python([0-9]+\.[0-9]+))")
            m = p.match(python_ld_flags)
            python_version = python_lib = None
            if m:
                python_lib = m.group(1)
                python_version = m.group(2)
            if python_version:
                logging.debug("Using system python: {0}".format(python_version))
                file_name = "libpython" + python_version + ".so"
                for path in "/usr/lib", python_lib, "/usr/local/lib", "/usr/lib64", "/usr/lib/x86_64-linux-gnu":
                    if os.path.exists(os.path.join(path, file_name)):
                        python_libs = os.path.join(path, file_name)
                        break

        python_dict["PYTHON_LIBRARIES"] = python_libs or "python-system-libs-not-found"
    return python_dict


# define Perl
def parse_perl_verbose(perl, param, tail=""):
    regex = "^{0}='(.*)'".format(param)
    p = re.compile(regex)
    try:
        cmd_result = get_stdout([perl, "-V:{0}".format(param)])
        if cmd_result:
            m = p.match(cmd_result)
            return m.group(1) + tail if m else None
    except FailedCmdException:
        return None


def init_dist_perl(host_os_id):
    perl_dict = {"PERL": "/usr/bin/perl", "YMAKE_PRINT_SPECS": ""}
    if host_os_id == "FREEBSD":
        perl_dict["PERL_VERSION"] = "51204"
        perl_dict["EXTUTILS"] = "/usr/local/lib/perl5/5.12.4/ExtUtils"
        perl_dict["PERLLIB"] = "/usr/local/lib/perl5/5.12.4/mach"
        perl_dict["PERLLIB_PATH"] = "/usr/local/lib/perl5/5.12.4/mach/CORE"
        perl_dict["PERLSITEARCHEXP"] = "/usr/local/lib/perl5/site_perl/5.12.4/mach"
        perl_dict["PERLINSTALLSITEARCH"] = "/usr/local/lib/perl5/site_perl/5.12.4/mach"
    elif host_os_id == "LINUX":
        perl_dict["PERL_VERSION"] = "51402"
        perl_dict["EXTUTILS"] = "/usr/share/perl/5.14/ExtUtils"
        perl_dict["PERLLIB"] = "/usr/lib/perl/5.14"
        perl_dict["PERLLIB_PATH"] = "/usr/lib/perl/5.14/CORE"
        perl_dict["PERLSITEARCHEXP"] = "/usr/local/lib/perl/5.14.2"
        perl_dict["PERLINSTALLSITEARCH"] = "/usr/local/lib/perl/5.14.2"
    else:
        if host_os_id == 'LINUX':
            perl_dict["WARNING"] = "Perl was not defined under this platform {0} for distbuild".format(host_os_id)
        perl_dict["PERL_VERSION"] = "perl-version-not-defined"
        perl_dict["EXTUTILS"] = "perl-privlibexp-not-found"
        perl_dict["PERLLIB"] = "perl-lib-not-found"
        perl_dict["PERLLIB_PATH"] = "perl-lib-CORE-not-found"
        perl_dict["PERLSITEARCHEXP"] = "perl-sitearchexp-not-found"
        perl_dict["PERLINSTALLSITEARCH"] = "perl-installsitearch-not-found"
    return perl_dict


def init_system_perl(host_os_id):
    perl_name = preset("PERL") or "perl"
    perl = which(perl_name)
    perl_dict = {"PERL": perl, "PERL_VERSION": 0, "YMAKE_PRINT_SPECS": ""}
    if perl:
        logging.debug("Using system perl: {0}".format(perl))
        perl_dict["PERL_VERSION"] = preset("PERL_VERSION") or convert_version(parse_perl_verbose(perl, "version")) or "perl-version-not-defined"
        perl_dict["EXTUTILS"] = preset("EXTUTILS") or parse_perl_verbose(perl, "privlibexp", "/ExtUtils") or "perl-privlibexp-not-found"
        perl_dict["PERLLIB"] = preset("PERLLIB") or parse_perl_verbose(perl, "archlib") or "perl-lib-not-found"
        perl_dict["PERLLIB_PATH"] = preset("PERLLIB_PATH") or parse_perl_verbose(perl, "archlib", "/CORE") or "perl-lib-CORE-not-found"
        perl_dict["PERLSITEARCHEXP"] = preset("PERLSITEARCHEXP") or parse_perl_verbose(perl, "sitearchexp") or "perl-sitearchexp-not-found"
        perl_dict["PERLINSTALLSITEARCH"] = preset("PERLINSTALLSITEARCH") or parse_perl_verbose(perl, "installsitearch") or "perl-installsitearch-not-found"

        if host_os_id == "WINDOWS":
            ld_perllib = preset("LD_PERLLIB") or parse_perl_verbose(perl, "libperl")
            for path in [ld_perllib, "perl512.lib", "perl510.lib", "perl58.lib"]:
                if os.path.exists(os.path.join(perl_dict["PERLLIB_PATH"], path)):
                    perl_dict["LD_PERLLIB"] = os.path.exists(os.path.join(perl_dict["PERLLIB_PATH"], path))
                    break
    return perl_dict


def find_perl(host_os_id):
    perl_dict = init_dist_perl(host_os_id) if not opts().mine_system_pathes else init_system_perl(host_os_id)
    perl_dict["LINK_STATIC_PERL"] = preset("LINK_STATIC_PERL") or "yes"
    perl_dict["PERLSUFFIX"] = "-csuffix .cpp"
    perl_dict["PERL_POLLUTE"] = perl_dict["PERLLIB_BIN"] = perl_dict["LD_PERLLIB"] = perl_dict["XSUBPPFLAGS"] = ""
    if is_positive("USE_PERL_5_6"):
        perl_dict["PERL_POLLUTE"] = "-DPERL_POLLUTE"
        perl_dict["YMAKE_PRINT_SPECS"] = """
USE_PERL_LIB {
    SET_APPEND(C_DEFINES $PERL_POLLUTE);
}"""
    return perl_dict


# define SVN(git)


def find_svn():
    pardir = ""
    for i in range(0, 3):
        for path in ['.svn/wc.db', '.svn/entries', '.git/HEAD']:
            if os.path.exists(os.path.join(ymake().arc_root, pardir, path)):
                return "${{input;hide:\"{0}\"}}".format(os.path.join(ymake().arc_root, pardir, path))
        pardir = os.path.join(pardir, os.pardir)

    return None


printed_other_settings = {}


def print_other_settings(build_sys, host_os_id):
    key = build_sys + "@" + host_os_id
    if key in printed_other_settings:
        return printed_other_settings[key]

    blk = BuildTypeBlock("name=Other.bs={0}.hOS={1}.bt=*".format(build_sys, host_os_id))
    if build_sys == "ymake" and host_os_id != l_system().os:  # XXX: deprecated. remove ASAP
        blk.add(" WARNING=Cross-platform local build of some stuff does not yet have configuration.")
        blk.write()
        return blk.name
    printed_other_settings[key] = blk.name
    print_conf_style(blk, find_perl(host_os_id))
    if host_os_id == "LINUX" or host_os_id == "FREEBSD" or host_os_id == "DARWIN" or host_os_id == "CYGWIN":
        nix_python = find_nix_python()
        if nix_python is not None:
            print_conf_style(blk, nix_python)

        if not is_positive("NO_SVN_DEPENDS") and opts().mine_system_pathes:
            blk.emit('SVN_DEPENDS', find_svn())
        else:
            blk.emit('SVN_DEPENDS')
    elif host_os_id == "WINDOWS":
        blk.emit('SVN_DEPENDS')
    blk.write()
    return blk.name


def reformat_env(env, values_sep=":"):
    def reformat_var(name, values):
        return "{}={}".format(name, ("\\" + values_sep).join(values))

    return " ".join("${env:\"" + reformat_var(name, values) + "\"}" for name, values in env.iteritems())


class Platform(object):
    def __init__(self, platform_json, params_json):
        self._os = platform_json['os']
        self._name = platform_json.get('visible_name', platform_json['toolchain'])
        self._arch = platform_json['arch'].lower()
        self.os = validate_os_name(self._os)
        self.arch = self._arch.lower()
        self.arch_opts = []
        self.environ = []
        if params_json:
            self.arch_opt = params_json['params'].get('arch_opt', [])
            self.environ = params_json.get('env', {})

    @property
    def is_intel(self):
        return self.arch in ('i386', 'i686', 'x86', 'x86_64')

    @property
    def is_arm(self):
        return self.arch == 'arm'

    @property
    def is_32_bit(self):
        return self.arch in ('i386', 'i686', 'x86', 'arm')

    @property
    def is_64_bit(self):
        return self.arch == 'x86_64'

    def __str__(self):
        return '{name}-{os}-{arch}'.format(name=self._name, os=self._os, arch=self._arch)


class BuildSettings(object):
    def __init__(self, params_json):
        self.params = params_json['params']
        self.build_type = opts().build_type.lower()
        self.host = Platform(params_json['platform']['host'], None)
        self.target = Platform(params_json['platform']['target'], params_json)

    @property
    def is_release(self):
        return self.build_type == 'release' or self.build_type.endswith('-release')

    @property
    def is_debug(self):
        return self.build_type == 'debug' or self.build_type.endswith('-debug')

    @property
    def is_ide(self):
        return self.build_type == IDE_BUILD_TYPE


def main():
    # arguments: ArcRoot, BuildType, Verbosity, path to local.ymake
    with open(YMake.find_core_conf(), 'r') as fin:
        print fin.read()

    if opts().presets:
        print '# Variables set from command line by -D options'
        for key in opts().presets:
            print '{0}={1}'.format(key, opts().presets[key])

    ymake().print_settings()

    ymake().print_build_type(opts().build_type)

    ymake().print_build_system(opts().build_system)

    ymake().print_pic("no" if is_positive("FORCE_NO_PIC") else "yes")

    tc_params = load_toolchain_params()
    build_settings = BuildSettings(tc_params)
    target_tc = "theyknow"  # this is hardcoded to trick yaconf()
    actual_bt = print_full_build_type(build_settings, target_tc, opts().build_system,
                                      validate_os_name(tc_params["platform"]["host"]["os"]),
                                      validate_os_name(tc_params["platform"]["target"]["os"]),
                                      tc_params["platform"]["target"]["arch"])
    BuildTypeBlock("bt=*.bs={0}.PIC=*".format(opts().build_system), actual_bt).write()

    print "# This block is what current ymake core is searching for"
    print "# This way, ymake will use these \"bt\" and \"bs\" regardless the stuff on its command line"
    BuildTypeBlock(
        "bt=*.bs=*", "bt={0}.bs={1}.PIC={2}".format(opts().build_type, opts().build_system, "no" if is_positive("FORCE_NO_PIC") else "yes")
    ).write()

    emit("CONF_SCRIPT_DEPENDS", __file__)


if __name__ == "__main__":
    main()
