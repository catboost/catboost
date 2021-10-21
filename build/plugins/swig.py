import os
import posixpath
import re

import _import_wrapper as iw
import _common as common


def init():
    iw.addrule('swg', Swig)


class Swig(iw.CustomCommand):
    def __init__(self, path, unit):
        self._tool = unit.get('SWIG_TOOL')
        self._library_dir = unit.get('SWIG_LIBRARY') or 'contrib/tools/swig/Lib'
        self._local_swig = unit.get('USE_LOCAL_SWIG') == "yes"

        self._path = path
        self._flags = ['-cpperraswarn']

        self._bindir = common.tobuilddir(unit.path())
        self._input_name = common.stripext(os.path.basename(self._path))

        relpath = os.path.relpath(os.path.dirname(self._path), unit.path())

        self._swig_lang = unit.get('SWIG_LANG')

        if self._swig_lang != 'jni_java':
            self._main_out = os.path.join(
                self._bindir,
                '' if relpath == '.' else relpath.replace('..', '__'),
                self._input_name + '_wrap.swg.c')

            if not path.endswith('.c.swg'):
                self._flags += ['-c++']
                self._main_out += 'pp'

        # lang_specific_incl_dir = 'perl5' if self._swig_lang == 'perl' else self._swig_lang
        lang_specific_incl_dir = self._swig_lang
        if self._swig_lang == 'perl':
            lang_specific_incl_dir = 'perl5'
        elif self._swig_lang in ['jni_cpp', 'jni_java']:
            lang_specific_incl_dir = 'java'
        incl_dirs = [
            "FOR", "swig",
            posixpath.join(self._library_dir, lang_specific_incl_dir),
            "FOR", "swig",
            self._library_dir
        ]
        self._incl_dirs = ['$S', '$B'] + [posixpath.join('$S', d) for d in incl_dirs]

        modname = unit.get('REALPRJNAME')
        self._flags.extend(['-module', modname])

        if not self._local_swig:
            unit.onaddincl(incl_dirs)

        if self._swig_lang == 'python':
            self._out_name = modname + '.py'
            self._flags.extend(['-interface', unit.get('MODULE_PREFIX') + modname])

        if self._swig_lang == 'perl':
            self._out_name = modname + '.pm'
            self._flags.append('-shadow')
            unit.onpeerdir(['build/platform/perl'])

        if self._swig_lang in ['jni_cpp', 'java']:
            self._out_header = os.path.splitext(self._main_out)[0] + '.h'
            if (not unit.get('USE_SYSTEM_JDK')) and (unit.get('OS_ANDROID') != "yes"):
                unit.onpeerdir(['contrib/libs/jdk'])

        self._package = 'ru.yandex.' + os.path.dirname(self._path).replace('$S/', '').replace('$B/', '').replace('/', '.').replace('-', '_')
        if self._swig_lang in ['jni_java', 'java']:
            self._out_name = os.path.splitext(os.path.basename(self._path))[0] + '.jsrc'
        elif self._swig_lang != 'jni_cpp':
            self._flags.append('-' + self._swig_lang)

    def descr(self):
        return 'SW', self._path, 'yellow'

    def flags(self):
        return self._flags

    def tools(self):
        return ['contrib/tools/swig'] if not self._tool else []

    def input(self):
        return [
            (self._path, [])
        ]

    def output(self):
        if self._swig_lang == 'jni_java':
            return [(common.join_intl_paths(self._bindir, self._out_name), [])]
        elif self._swig_lang == 'jni_cpp':
            return [(self._main_out, []), (self._out_header, [])]

        return [
            (self._main_out, []),
            (common.join_intl_paths(self._bindir, self._out_name), (['noauto', 'add_to_outs'] if self._swig_lang != 'java' else [])),
        ] + ([(self._out_header, [])] if self._swig_lang == 'java' else [])

    def output_includes(self):
        return [(self._out_header, [])] if self._swig_lang in ['java', 'jni_cpp'] else []

    def run(self, extra_args, binary):
        if self._local_swig:
            binary = self._tool
        return self.do_run_java(binary, self._path) if self._swig_lang in ['java', 'jni_cpp', 'jni_java'] else self.do_run(binary, self._path)

    def _incl_flags(self):
        return ['-I' + self.resolve_path(x) for x in self._incl_dirs]

    def do_run(self, binary, path):
        self.call([binary] + self._flags + [
            '-o', self.resolve_path(common.get(self.output, 0)),
            '-outdir', self.resolve_path(self._bindir)
        ] + self._incl_flags() + [self.resolve_path(path)])

    def do_run_java(self, binary, path):
        import tarfile

        outdir = self.resolve_path(self._bindir)
        if self._swig_lang != 'jni_cpp':
            java_srcs_dir = os.path.join(outdir, self._package.replace('.', '/'))
            if not os.path.exists(java_srcs_dir):
                os.makedirs(java_srcs_dir)

        flags = self._incl_flags()
        src = self.resolve_path(path)
        with open(src, 'r') as f:
            if not re.search(r'(?m)^%module\b', f.read()):
                flags += ['-module', os.path.splitext(os.path.basename(src))[0]]

        if self._swig_lang == 'jni_cpp':
            self.call([binary, '-c++', '-o', self._main_out, '-java', '-package', self._package] + flags + [src])
        elif self._swig_lang == 'jni_java':
            self.call([binary, '-c++', '-o', os.path.join(outdir, 'unused.cpp'), '-outdir', java_srcs_dir, '-java', '-package', self._package] + flags + [src])
        elif self._swig_lang == 'java':
            self.call([
                binary, '-c++', '-o', self._main_out, '-outdir', java_srcs_dir,
                '-java', '-package', self._package,
            ] + flags + [src])

        if self._swig_lang in ['jni_java', 'java']:
            with tarfile.open(os.path.join(outdir, self._out_name), 'a') as tf:
                tf.add(java_srcs_dir, arcname=self._package.replace('.', '/'))

        if self._swig_lang in ['jni_cpp', 'java']:
            header = os.path.splitext(self.resolve_path(self._main_out))[0] + '.h'
            if not os.path.exists(header):
                open(header, 'w').close()


def on_swig_lang_filtered_srcs(unit, *args):
    swig_lang = unit.get('SWIG_LANG')
    allowed_exts = set()
    if swig_lang == 'jni_cpp':
        allowed_exts = set(['.cpp', '.swg'])
    if swig_lang == 'jni_java':
        allowed_exts = set(['.java', '.swg'])
    args = [arg for arg in iter(args) if allowed_exts and os.path.splitext(arg)[1] in allowed_exts]
    unit.onsrcs(args)
