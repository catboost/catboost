import __res


def repl():
    user_ns = {}
    py_main = __res.find('PY_MAIN')

    if py_main:
        mod_name, func_name = py_main.split(b':', 1)
        try:
            import importlib
            mod = importlib.import_module(mod_name.decode('UTF-8'))
            user_ns = mod.__dict__
        except:
            import traceback
            traceback.print_exc()

    if py_main and '__main__' not in user_ns:
        def run(args):
            if isinstance(args, basestring):
                import shlex
                args = shlex.split(args)

            import sys
            sys.argv = [sys.argv[0]] + args
            getattr(mod, func_name)()

        user_ns['__main__'] = run

    try:
        import IPython
    except ModuleNotFoundError:
        pass
    else:
        return IPython.start_ipython(user_ns=user_ns)

    import code
    code.interact(local=user_ns)
