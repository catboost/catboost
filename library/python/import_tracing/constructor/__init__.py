import os


def init():
    if "Y_PYTHON_TRACE_FILE" in os.environ:
        import atexit
        import library.python.import_tracing.lib.regulator as regulator

        regulator.enable(os.getenv("Y_PYTHON_TRACE_FILE"))
        atexit.register(regulator.disable)
