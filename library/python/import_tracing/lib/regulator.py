import collections
import os

_Instance = collections.namedtuple("_Instance", ("import_tracer", "converter", "filepath"))

INSTANCE = None


def _get_converter_instance():
    import library.python.import_tracing.lib.converters.raw as text_converter
    import library.python.import_tracing.lib.converters.chrometrace as chrome_converter

    converter_mapping = {"text": text_converter.RawTextTraceConverter, "evlog": chrome_converter.ChromiumTraceConverter}

    env_val = os.getenv("Y_PYTHON_TRACE_FORMAT")

    converter = converter_mapping.get(env_val, text_converter.RawTextTraceConverter)

    return converter()


def _resolve_filepath(filemask):
    import socket
    import sys

    pid = os.getpid()
    hostname = socket.gethostname()
    executable_filename = os.path.basename(sys.executable)

    return filemask.replace("%p", str(pid)).replace("%h", hostname).replace("%e", executable_filename)


def enable(filemask):
    import library.python.import_tracing.lib.import_tracer as import_tracer
    import __res

    global INSTANCE

    if INSTANCE is not None:
        return INSTANCE

    converter = _get_converter_instance()
    import_tracer = import_tracer.ImportTracer()

    def before_import_callback(modname, filename):
        import_tracer.start_event(modname, filename)

    def after_import_callback(modname, filename):
        import_tracer.finish_event(modname, filename)

    __res.importer.set_callbacks(before_import_callback, after_import_callback)

    filepath = _resolve_filepath(filemask)

    new_instance = _Instance(import_tracer, converter, filepath)
    INSTANCE = new_instance

    return new_instance


def disable(close_not_finished=False):
    global INSTANCE

    if INSTANCE is None:
        return

    import_tracer = INSTANCE.import_tracer
    converter = INSTANCE.converter
    filepath = INSTANCE.filepath

    converter.dump(import_tracer.get_events(close_not_finished), filepath)

    INSTANCE = None
