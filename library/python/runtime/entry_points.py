import sys

import __res


def repl():
    user_ns = {}
    py_main = __res.find('PY_MAIN')

    if py_main:
        py_main_split = py_main.split(':', 1)
        if len(py_main_split) == 2:
            mod_name, func_name = py_main_split
        else:
            mod_name, func_name = py_main_split[0], 'main'

        if not mod_name:
            mod_name = 'library.python.runtime.entry_points'

        try:
            import importlib
            mod = importlib.import_module(mod_name)
            user_ns = mod.__dict__
        except:  # noqa E722
            import traceback
            traceback.print_exc()

        if '__main__' not in user_ns:
            def run(args):
                if isinstance(args, basestring):
                    import shlex
                    args = shlex.split(args)

                import sys
                sys.argv = [sys.argv[0]] + args
                getattr(mod, func_name)()

            user_ns['__main__'] = run
    else:
        try:
            mod = __res.importer.load_module('__main__', fix_name='__main_real')
            user_ns = mod.__dict__
        except ImportError:
            pass

    try:
        import IPython
    except ImportError:
        import code
        code.interact(local=user_ns)
    else:
        IPython.start_ipython(user_ns=user_ns)


def resource_files():
    sys.stdout.write('\n'.join(sorted(__res.resfs_files()) + ['']))


def run_constructors():
    for key, module_name in __res.iter_keys('py/constructors/'):
        import importlib
        module = importlib.import_module(module_name)
        init_func = getattr(module, __res.find(key))
        init_func()
