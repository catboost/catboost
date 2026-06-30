import sys

import __res


def repl():
    user_ns = {}
    py_main = __res.find("PY_MAIN")

    if py_main:
        mod_name, func_name = (py_main.split(b":", 1) + [None])[:2]
        try:
            import importlib

            mod = importlib.import_module(mod_name.decode("UTF-8"))
            user_ns = mod.__dict__
        except ModuleNotFoundError:
            import traceback

            traceback.print_exc()

        if func_name and "__main__" not in user_ns:

            def run(args):
                if isinstance(args, str):
                    import shlex

                    args = shlex.split(args)

                import sys

                sys.argv = [sys.argv[0]] + args
                getattr(mod, func_name)()

            user_ns["__main__"] = run

    try:
        import IPython
    except ModuleNotFoundError:
        pass
    else:
        return IPython.start_ipython(user_ns=user_ns)

    import code

    code.interact(local=user_ns)


def resource_files():
    sys.stdout.buffer.write(b"\n".join(sorted(__res.resfs_files()) + [b""]))


def run_constructors():
    for key, module_name in __res.iter_keys(b"py/constructors/"):
        import importlib

        module = importlib.import_module(module_name.decode())
        init_func = getattr(module, __res.find(key).decode())
        init_func()
