PY3TEST()



PEERDIR(
    contrib/python/parso
)

DATA(
    arcadia/contrib/python/parso/py3/tests
)

TEST_SRCS(
    __init__.py
    conftest.py
    failing_examples.py
    test_cache.py
    test_diff_parser.py
    test_dump_tree.py
    test_error_recovery.py
    test_file_python_errors.py
    test_fstring.py
    test_get_code.py
    test_grammar.py
    test_load_grammar.py
    test_normalizer_issues_files.py
    test_old_fast_parser.py
    test_param_splitting.py
    test_parser.py
    test_parser_tree.py
    test_pep8.py
    test_pgen2.py
    test_prefix.py
    test_python_errors.py
    test_tokenize.py
    test_utils.py
)

NO_LINT()

END()
