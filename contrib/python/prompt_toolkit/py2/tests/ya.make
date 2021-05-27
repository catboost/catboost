PY2_LIBRARY()

LICENSE(BSD-3-Clause)



PEERDIR(
    contrib/python/prompt_toolkit/py2
)

TEST_SRCS(
    test_buffer.py
    test_cli.py
    test_contrib.py
    test_document.py
    test_filter.py
    test_inputstream.py
    test_key_binding.py
    test_layout.py
    test_print_tokens.py
    test_regular_languages.py
    test_shortcuts.py
    test_style.py
    test_utils.py
    test_yank_nth_arg.py
)

NO_LINT()

END()

RECURSE_FOR_TESTS(
    py2
)
