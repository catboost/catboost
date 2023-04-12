"""
A Cython plugin for coverage.py

Requires the coverage package at least in version 4.0 (which added the plugin API).
"""

from __future__ import absolute_import

import re
import os.path
import sys
from collections import defaultdict

from coverage.plugin import CoveragePlugin, FileTracer, FileReporter  # requires coverage.py 4.0+
from coverage.files import canonical_filename

from .Utils import find_root_package_dir, is_package_dir, open_source_file


from . import __version__


C_FILE_EXTENSIONS = ['.c', '.cpp', '.cc', '.cxx']
MODULE_FILE_EXTENSIONS = set(['.py', '.pyx', '.pxd'] + C_FILE_EXTENSIONS)


def _find_c_source(base_path):
    file_exists = os.path.exists
    for ext in C_FILE_EXTENSIONS:
        file_name = base_path + ext
        if file_exists(file_name):
            return file_name
    return None


def _find_dep_file_path(main_file, file_path, relative_path_search=False):
    abs_path = os.path.abspath(file_path)
    if not os.path.exists(abs_path) and (file_path.endswith('.pxi') or
                                         relative_path_search):
        # files are looked up relative to the main source file
        rel_file_path = os.path.join(os.path.dirname(main_file), file_path)
        if os.path.exists(rel_file_path):
            abs_path = os.path.abspath(rel_file_path)
    # search sys.path for external locations if a valid file hasn't been found
    if not os.path.exists(abs_path):
        for sys_path in sys.path:
            test_path = os.path.realpath(os.path.join(sys_path, file_path))
            if os.path.exists(test_path):
                return canonical_filename(test_path)
    return canonical_filename(abs_path)


class Plugin(CoveragePlugin):
    # map from traced file paths to absolute file paths
    _file_path_map = None
    # map from traced file paths to corresponding C files
    _c_files_map = None
    # map from parsed C files to their content
    _parsed_c_files = None

    def sys_info(self):
        return [('Cython version', __version__)]

    def file_tracer(self, filename):
        """
        Try to find a C source file for a file path found by the tracer.
        """
        # TODO We need to pxd-files to the include map. For more info see pybuild.py
        # Currently skip such files, because they are not supported in Arcadia pybuild with coverage.
        if os.path.splitext(filename)[-1] not in ('.pyx', '.pxi'):
            return None
        if filename.startswith('<') or filename.startswith('memory:'):
            return None
        c_file = py_file = None
        filename = canonical_filename(filename)
        if self._c_files_map and filename in self._c_files_map:
            c_file = self._c_files_map[filename][0]

        if c_file is None:
            c_file, py_file = self._find_source_files(filename)
            if not c_file:
                return None  # unknown file

            # parse all source file paths and lines from C file
            # to learn about all relevant source files right away (pyx/pxi/pxd)
            # FIXME: this might already be too late if the first executed line
            #        is not from the main .pyx file but a file with a different
            #        name than the .c file (which prevents us from finding the
            #        .c file)
            _, code = self._read_source_lines(c_file, filename)
            if code is None:
                return None  # no source found

        if self._file_path_map is None:
            self._file_path_map = {}
        return CythonModuleTracer(filename, py_file, c_file, self._c_files_map, self._file_path_map)

    def file_reporter(self, filename):
        # TODO: let coverage.py handle .py files itself
        #ext = os.path.splitext(filename)[1].lower()
        #if ext == '.py':
        #    from coverage.python import PythonFileReporter
        #    return PythonFileReporter(filename)

        filename = canonical_filename(filename)
        if self._c_files_map and filename in self._c_files_map:
            c_file, rel_file_path, code = self._c_files_map[filename]
        else:
            c_file, _ = self._find_source_files(filename)
            if not c_file:
                if standalone():
                    raise AssertionError(filename)
                return None  # unknown file
            rel_file_path, code = self._read_source_lines(c_file, filename)
            if code is None:
                if standalone():
                    raise AssertionError(filename)
                return None  # no source found

        return CythonModuleReporter(c_file, filename, rel_file_path, code)

    def _find_source_files(self, filename):
        basename, ext = os.path.splitext(filename)
        ext = ext.lower()
        if ext in MODULE_FILE_EXTENSIONS:
            pass
        elif ext == '.pyd':
            # Windows extension module
            platform_suffix = re.search(r'[.]cp[0-9]+-win[_a-z0-9]*$', basename, re.I)
            if platform_suffix:
                basename = basename[:platform_suffix.start()]
        elif ext == '.so':
            # Linux/Unix/Mac extension module
            platform_suffix = re.search(r'[.](?:cpython|pypy)-[0-9]+[-_a-z0-9]*$', basename, re.I)
            if platform_suffix:
                basename = basename[:platform_suffix.start()]
        elif ext == '.pxi':
            # if we get here, it means that the first traced line of a Cython module was
            # not in the main module but in an include file, so try a little harder to
            # find the main source file
            self._find_c_source_files(os.path.dirname(filename), filename)
            if filename in self._c_files_map:
                return self._c_files_map[filename][0], None
            if standalone():
                raise AssertionError(filename)
        else:
            # none of our business
            return None, None

        c_file = filename if ext in C_FILE_EXTENSIONS else _find_c_source(basename)
        if c_file is None:
            # a module "pkg/mod.so" can have a source file "pkg/pkg.mod.c"
            package_root = find_root_package_dir.uncached(filename)
            package_path = os.path.relpath(basename, package_root).split(os.path.sep)
            if len(package_path) > 1:
                test_basepath = os.path.join(os.path.dirname(filename), '.'.join(package_path))
                c_file = _find_c_source(test_basepath)

        py_source_file = None
        if c_file:
            py_source_file = os.path.splitext(c_file)[0] + '.py'
            if not os.path.exists(py_source_file):
                py_source_file = None

            try:
                with OpenFile(c_file) as f:
                    if '/* Generated by Cython ' not in f.read(30):
                        return None, None  # not a Cython file
            except (IOError, OSError):
                c_file = None

        return c_file, py_source_file

    def _find_c_source_files(self, dir_path, source_file):
        """
        Desperately parse all C files in the directory or its package parents
        (not re-descending) to find the (included) source file in one of them.
        """
        if standalone():
            if os.environ.get('PYTHON_COVERAGE_CYTHON_BUILD_ROOT'):
                broot = os.environ['PYTHON_COVERAGE_CYTHON_BUILD_ROOT']
                iter_files = lambda: (os.path.join(root, filename) for root, _, files in os.walk(broot) for filename in files)
            else:
                import library.python.resource
                iter_files = library.python.resource.resfs_files
            for c_file in iter_files():
                if os.path.splitext(c_file)[1] in C_FILE_EXTENSIONS:
                    self._read_source_lines(c_file, source_file)
                    if source_file in self._c_files_map:
                        return
            raise AssertionError((source_file, os.environ.get('PYTHON_COVERAGE_CYTHON_BUILD_ROOT')))

        if not os.path.isdir(dir_path):
            return
        splitext = os.path.splitext
        for filename in os.listdir(dir_path):
            ext = splitext(filename)[1].lower()
            if ext in C_FILE_EXTENSIONS:
                self._read_source_lines(os.path.join(dir_path, filename), source_file)
                if source_file in self._c_files_map:
                    return
        # not found? then try one package up
        if is_package_dir(dir_path):
            self._find_c_source_files(os.path.dirname(dir_path), source_file)

    def _read_source_lines(self, c_file, sourcefile):
        """
        Parse a Cython generated C/C++ source file and find the executable lines.
        Each executable line starts with a comment header that states source file
        and line number, as well as the surrounding range of source code lines.
        """
        if self._parsed_c_files is None:
            self._parsed_c_files = {}
        if c_file in self._parsed_c_files:
            code_lines = self._parsed_c_files[c_file]
        else:
            code_lines = self._parse_cfile_lines(c_file)
            self._parsed_c_files[c_file] = code_lines

        if self._c_files_map is None:
            self._c_files_map = {}

        for filename, code in code_lines.items():
            abs_path = _find_dep_file_path(c_file, filename,
                                           relative_path_search=True)
            self._c_files_map[abs_path] = (c_file, filename, code)

        if sourcefile not in self._c_files_map:
            return (None,) * 2  # e.g. shared library file
        return self._c_files_map[sourcefile][1:]

    def _parse_cfile_lines(self, c_file):
        """
        Parse a C file and extract all source file lines that generated executable code.
        """
        match_source_path_line = re.compile(r' */[*] +"(.*)":([0-9]+)$').match
        match_current_code_line = re.compile(r' *[*] (.*) # <<<<<<+$').match
        match_comment_end = re.compile(r' *[*]/$').match
        match_trace_line = re.compile(r' *__Pyx_TraceLine\(([0-9]+),').match
        not_executable = re.compile(
            r'\s*c(?:type)?def\s+'
            r'(?:(?:public|external)\s+)?'
            r'(?:struct|union|enum|class)'
            r'(\s+[^:]+|)\s*:'
        ).match

        code_lines = defaultdict(dict)
        executable_lines = defaultdict(set)
        current_filename = None

        with OpenFile(c_file) as lines:
            lines = iter(lines)
            for line in lines:
                match = match_source_path_line(line)
                if not match:
                    if '__Pyx_TraceLine(' in line and current_filename is not None:
                        trace_line = match_trace_line(line)
                        if trace_line:
                            executable_lines[current_filename].add(int(trace_line.group(1)))
                    continue
                filename, lineno = match.groups()
                current_filename = filename
                lineno = int(lineno)
                for comment_line in lines:
                    match = match_current_code_line(comment_line)
                    if match:
                        code_line = match.group(1).rstrip()
                        if not_executable(code_line):
                            break
                        code_lines[filename][lineno] = code_line
                        break
                    elif match_comment_end(comment_line):
                        # unexpected comment format - false positive?
                        break

        # Remove lines that generated code but are not traceable.
        for filename, lines in code_lines.items():
            dead_lines = set(lines).difference(executable_lines.get(filename, ()))
            for lineno in dead_lines:
                del lines[lineno]
        return code_lines


class CythonModuleTracer(FileTracer):
    """
    Find the Python/Cython source file for a Cython module.
    """
    def __init__(self, module_file, py_file, c_file, c_files_map, file_path_map):
        super(CythonModuleTracer, self).__init__()
        self.module_file = module_file
        self.py_file = py_file
        self.c_file = c_file
        self._c_files_map = c_files_map
        self._file_path_map = file_path_map

    def has_dynamic_source_filename(self):
        return True

    def dynamic_source_filename(self, filename, frame):
        """
        Determine source file path.  Called by the function call tracer.
        """
        source_file = frame.f_code.co_filename
        try:
            return self._file_path_map[source_file]
        except KeyError:
            pass
        if standalone():
            abs_path = self.module_file
        else:
            abs_path = _find_dep_file_path(filename, source_file)

        if self.py_file and source_file[-3:].lower() == '.py':
            # always let coverage.py handle this case itself
            self._file_path_map[source_file] = self.py_file
            return self.py_file

        assert self._c_files_map is not None
        if abs_path not in self._c_files_map:
            self._c_files_map[abs_path] = (self.c_file, source_file, None)
        self._file_path_map[source_file] = abs_path
        return abs_path


class CythonModuleReporter(FileReporter):
    """
    Provide detailed trace information for one source file to coverage.py.
    """
    def __init__(self, c_file, source_file, rel_file_path, code):
        super(CythonModuleReporter, self).__init__(source_file)
        self.name = rel_file_path
        self.c_file = c_file
        self._code = code
        self._abs_filename = self._find_abs_filename()

    def lines(self):
        """
        Return set of line numbers that are possibly executable.
        """
        return set(self._code)

    def _iter_source_tokens(self):
        current_line = 1
        for line_no, code_line in sorted(self._code.items()):
            while line_no > current_line:
                yield []
                current_line += 1
            yield [('txt', code_line)]
            current_line += 1

    def source(self):
        """
        Return the source code of the file as a string.
        """
        if os.path.exists(self._abs_filename):
            with open_source_file(self._abs_filename) as f:
                return f.read()
        else:
            return '\n'.join(
                (tokens[0][1] if tokens else '')
                for tokens in self._iter_source_tokens())

    def source_token_lines(self):
        """
        Iterate over the source code tokens.
        """
        if os.path.exists(self._abs_filename):
            with open_source_file(self._abs_filename) as f:
                for line in f:
                    yield [('txt', line.rstrip('\n'))]
        else:
            for line in self._iter_source_tokens():
                yield line

    def _find_abs_filename(self):
        for root in [
            os.environ.get('PYTHON_COVERAGE_ARCADIA_SOURCE_ROOT'),
            os.environ.get('PYTHON_COVERAGE_CYTHON_BUILD_ROOT'),
        ]:
            if root:
                abs_path = os.path.join(root, self.filename)
                if root and os.path.exists(abs_path):
                    return abs_path
        return self.filename


def coverage_init(reg, options):
    reg.add_file_tracer(Plugin())


# ========================== Arcadia specific =================================

def standalone():
    return getattr(sys, 'is_standalone_binary', False)


class OpenFile(object):

    def __init__(self, filename, mode='r'):
        assert 'r' in mode, ('Read-only', mode)
        self.filename = filename
        self.mode = mode
        self.file = None
        self.build_root = os.environ.get('PYTHON_COVERAGE_CYTHON_BUILD_ROOT')

    def __enter__(self):
        # See redefined _find_c_source() description for more info
        if self.build_root:
            self.file = open(os.path.join(self.build_root, self.filename), self.mode)
            return self.file
        elif standalone():
            import library.python.resource
            from six import StringIO

            content = library.python.resource.resfs_read(self.filename, builtin=True)
            assert content, (self.filename, os.environ.items())
            return StringIO(content.decode())
        else:
            self.file = open(self.filename, self.mode)
            return self.file

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.file:
            self.file.close()

# ======================= Redefine some methods ===============================

if standalone():
    import itertools
    import json

    CYTHON_INCLUDE_MAP = {'undef': True}


    def _find_c_source(base_path):
        '''
        There are two different coverage stages when c source file might be required:
         * trace - python calls c_tracefunc on every line and CythonModuleTracer needs to match
            pyd and pxi files with source files. This is test's runtime and tests' clean environment might
            doesn't contain required sources and generated files (c, cpp), that's why we get files from resfs_src.
         * report - coverage data contains only covered data and CythonModuleReporter needs to
            parse source files to obtain missing lines and branches. This is test_tool's resolve/build_report step.
            test_tools doesn't have compiled in sources, however, it can extract required files
            from binary and set PYTHON_COVERAGE_CYTHON_BUILD_ROOT to guide coverage.
        '''
        if os.environ.get('PYTHON_COVERAGE_CYTHON_BUILD_ROOT'):
            # Report stage (resolve)
            def exists(filename):
                return os.path.exists(os.path.join(os.environ['PYTHON_COVERAGE_CYTHON_BUILD_ROOT'], filename))
        else:
            # Trace stage (test's runtime)
            def exists(filename):
                import library.python.resource
                return library.python.resource.resfs_src(filename, resfs_file=True)

        if os.environ.get('PYTHON_COVERAGE_CYTHON_INCLUDE_MAP'):
            if CYTHON_INCLUDE_MAP.get('undef'):
                with open(os.environ['PYTHON_COVERAGE_CYTHON_INCLUDE_MAP']) as afile:
                    data = json.load(afile)
                    data = {os.path.splitext(k)[0]: v for k, v in data.items()}

                CYTHON_INCLUDE_MAP.clear()
                CYTHON_INCLUDE_MAP.update(data)

            if base_path in CYTHON_INCLUDE_MAP:
                # target file was included and should be sought inside another pyx file
                base_path = CYTHON_INCLUDE_MAP[base_path]

        # TODO (', '.py3', '.py2') -> ('.py3', '.py2'), when https://a.yandex-team.ru/review/3511262 is merged
        suffixes = [''.join(x) for x in itertools.product(('.pyx',), ('', '.py3', '.py2'), ('.cpp', '.c'))]
        suffixes += C_FILE_EXTENSIONS

        for suffix in suffixes:
            if exists(base_path + suffix):
                return base_path + suffix

        return None


    def _find_dep_file_path(main_file, file_path, relative_path_search=False):
        # file_path is already arcadia root relative
        return canonical_filename(file_path)
