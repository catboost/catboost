PY3TEST()



PEERDIR(
    contrib/python/prompt_toolkit/py3
)

TEST_SRCS(
    test_async_generator.py
    test_buffer.py
    test_cli.py
    test_completion.py
    test_document.py
    test_filter.py
    test_formatted_text.py
    test_history.py
    test_inputstream.py
    test_key_binding.py
    test_layout.py
    test_print_formatted_text.py
    test_regular_languages.py
    test_shortcuts.py
    test_style.py
    test_style_transformation.py
    test_utils.py
    test_vt100_output.py
    test_widgets.py
    test_yank_nth_arg.py
)

NO_LINT()

SIZE(MEDIUM)

END()
