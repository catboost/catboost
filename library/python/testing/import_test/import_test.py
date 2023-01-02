from __future__ import print_function

import os
import re
import sys
import time
import signal
import traceback

import __res
from __res import importer


def setup_test_environment():
    try:
        from yatest_lib.ya import Ya
        import yatest.common as yc
        yc.runtime._set_ya_config(ya=Ya())
    except ImportError:
        pass


def check_imports(no_check=(), extra=(), skip_func=None, py_main=None):
    """
    tests all bundled modules are importable
    just add
    "PEERDIR(library/python/import_test)" to your CMakeLists.txt and
    "from import_test import test_imports" to your python test source file.
    """

    str_ = lambda s: s
    if not isinstance(b'', str):
        str_ = lambda s: s.decode('UTF-8')

    exceptions = list(no_check)
    for key, _ in __res.iter_keys(b'py/no_check_imports/'):
        exceptions += str_(__res.find(key)).split()
    if exceptions:
        exceptions.sort()
        print('NO_CHECK_IMPORTS', ' '.join(exceptions))

    patterns = [re.escape(s).replace(r'\*', r'.*') for s in exceptions]
    rx = re.compile('^({})$'.format('|'.join(patterns)))

    failed = []
    import_times = {}

    norm = lambda s: s[:-9] if s.endswith('.__init__') else s

    modules = sys.extra_modules | set(extra)
    modules = sorted(modules, key=norm)
    if py_main:
        modules = [py_main] + modules

    for module in modules:
        if module not in extra and (rx.search(module) or skip_func and skip_func(module)):
            print('SKIP', module)
            continue

        name = module.rsplit('.', 1)[-1]
        if name == '__main__' and 'if __name__ ==' not in importer.get_source(module):
            print('SKIP', module, '''without "if __name__ == '__main__'" check''')
            continue

        def print_backtrace_marked(e):
            tb_exc = traceback.format_exception(*e)
            for item in tb_exc:
                for l in item.splitlines():
                    print('FAIL:', l, file=sys.stderr)

        try:
            print('TRY', module)
            # XXX waiting for py3 to use print(..., flush=True)
            sys.stdout.flush()

            s = time.time()
            if module == '__main__':
                importer.load_module('__main__', '__main__py')
            elif module.endswith('.__init__'):
                __import__(module[:-len('.__init__')])
            else:
                __import__(module)

            delay = time.time() - s
            import_times[str(module)] = delay
            print('OK ', module, '{:.3f}s'.format(delay))

        except Exception as e:
            print('FAIL:', module, e, file=sys.stderr)
            print_backtrace_marked(sys.exc_info())
            failed.append('{}: {}'.format(module, e))

        except:
            e = sys.exc_info()
            print('FAIL:', module, e, file=sys.stderr)
            print_backtrace_marked(e)
            failed.append('{}: {}'.format(module, e))
            raise

    print("Slowest imports:")
    for m, t in sorted(import_times.items(), key=lambda x: x[1], reverse=True)[:30]:
        print('  ', '{:.3f}s'.format(t), m)

    if failed:
        raise ImportError('modules not imported:\n' + '\n'.join(failed))


test_imports = check_imports


def main():
    setup_test_environment()

    skip_names = sys.argv[1:]

    try:
        import faulthandler
    except ImportError:
        faulthandler = None

    if faulthandler:
        # Dump python backtrace in case of any errors
        faulthandler.enable()
        if hasattr(signal, "SIGUSR2"):
            # SIGUSR2 is used by test_tool to teardown tests
            faulthandler.register(signal.SIGUSR2, chain=True)

    os.environ['Y_PYTHON_IMPORT_TEST'] = ''

    # We should initialize Django before importing any applications
    if os.getenv('DJANGO_SETTINGS_MODULE'):
        try:
            import django
        except ImportError:
            pass
        else:
            django.setup()

    py_main = __res.find('PY_MAIN')

    if py_main:
        py_main_module = py_main.split(b':', 1)[0].decode('UTF-8')
    else:
        py_main_module = None

    try:
        check_imports(no_check=skip_names, py_main=py_main_module)
    except:
        sys.exit(1)
