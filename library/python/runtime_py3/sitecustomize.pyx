import traceback

# Use pure-python implementation of traceback printer.
# Built-in printer (PyTraceBack_Print) does not support custom module loaders
sys.excepthook = traceback.print_exception
