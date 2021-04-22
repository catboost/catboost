PY2_LIBRARY() # Need for Python 2 only

LICENSE(MIT)



VERSION(0.5)

PEERDIR(
    library/python/symbols/win_unicode_console
)

PY_SRCS(
    TOP_LEVEL
    win_unicode_console/__init__.py
    win_unicode_console/buffer.py
    win_unicode_console/console.py
    win_unicode_console/file_object.py
    win_unicode_console/info.py
    win_unicode_console/raw_input.py
    win_unicode_console/readline_hook.py
    win_unicode_console/runner.py
    win_unicode_console/streams.py
    win_unicode_console/tokenize_open.py
    win_unicode_console/unicode_argv.py
)

NO_LINT()

END()
