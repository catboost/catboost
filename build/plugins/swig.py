import os
import re

import _import_wrapper as iw
import _common as common


def init():
    iw.addrule('swg', Swig)


class Swig(iw.CustomCommand):
    def __init__(self, path, unit):
        self._path = path
        self._flags = ['-cpperraswarn']

        self._bindir = common.tobuilddir(unit.path())
        self._input_name = common.stripext(os.path.basename(self._path))

        relpath = os.path.relpath(os.path.dirname(self._path), unit.path())
        self._main_out = os.path.join(
            self._bindir,
            '' if relpath == '.' else relpath.replace('..', '__'),
            self._input_name + '_wrap.c')

        if not path.endswith('.c.swg'):
            self._flags += ['-c++']
            self._main_out += 'pp'

        self._swig_lang = unit.get('SWIG_LANG')

        lang_specific_incl_dir = 'perl5' if self._swig_lang == 'perl' else self._swig_lang
        incl_dirs = [
            'contrib/tools/swig/Lib/' + lang_specific_incl_dir,
            'contrib/tools/swig/Lib',
        ]
        self._incl_dirs = ['$S', '$B'] + ['$S/{}'.format(d) for d in incl_dirs]

        modname = unit.get('REALPRJNAME')
        self._flags.extend(['-module', modname])

        unit.onaddincl(incl_dirs)

        if self._swig_lang == 'python':
            self._out_name = modname + '.py'
            self._flags.extend(['-interface', unit.get('MODULE_PREFIX') + modname])

        if self._swig_lang == 'perl':
            self._out_name = modname + '.pm'
            self._flags.append('-shadow')
            unit.onpeerdir(['build/platform/perl'])

        if self._swig_lang == 'java':
            self._out_name = os.path.splitext(os.path.basename(self._path))[0] + '.jsrc'
            self._out_header = os.path.splitext(self._main_out)[0] + '.h'
            self._package = 'ru.yandex.' + os.path.dirname(self._path).replace('$S/', '').replace('$B/', '').replace('/', '.').replace('-', '_')
            if unit.get('OS_ANDROID') != "yes":
                unit.onpeerdir(['contrib/libs/jdk'])

        self._flags.append('-' + self._swig_lang)

    def descr(self):
        return 'SW', self._path, 'yellow'

    def flags(self):
        return self._flags

    def tools(self):
        return ['contrib/tools/swig']

    def input(self):
        return [
            (self._path, [])
        ]

    def output(self):
        return [
            (self._main_out, []),
            (common.join_intl_paths(self._bindir, self._out_name), (['noauto', 'add_to_outs'] if self._swig_lang != 'java' else [])),
        ] + ([(self._out_header, [])] if self._swig_lang == 'java' else [])

    def output_includes(self):
        return [(self._out_header, [])] if self._swig_lang == 'java' else []

    def run(self, extra_args, binary):
        return self.do_run(binary, self._path) if self._swig_lang != 'java' else self.do_run_java(binary, self._path)

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
        java_srcs_dir = os.path.join(outdir, self._package.replace('.', '/'))
        if not os.path.exists(java_srcs_dir):
            os.makedirs(java_srcs_dir)

        flags = self._incl_flags()
        src = self.resolve_path(path)
        with open(src, 'r') as f:
            if not re.search(r'(?m)^%module\b', f.read()):
                flags += ['-module', os.path.splitext(os.path.basename(src))[0]]

        self.call([
            binary, '-c++', '-o', self._main_out, '-outdir', java_srcs_dir,
            '-java', '-package', self._package,
        ] + flags + [src])

        with tarfile.open(os.path.join(outdir, self._out_name), 'a') as tf:
            tf.add(java_srcs_dir, arcname=self._package.replace('.', '/'))

        header = os.path.splitext(self.resolve_path(self._main_out))[0] + '.h'
        if not os.path.exists(header):
            open(header, 'w').close()
