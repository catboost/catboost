PY2_LIBRARY()

LICENSE(PSF-2.0)



VERSION(0.7.1)

PY_SRCS(
    TOP_LEVEL
    parso/__init__.py
    parso/_compatibility.py
    parso/cache.py
    parso/file_io.py
    parso/grammar.py
    parso/normalizer.py
    parso/parser.py
    parso/pgen2/__init__.py
    parso/pgen2/generator.py
    parso/pgen2/grammar_parser.py
    parso/python/__init__.py
    parso/python/diff.py
    parso/python/errors.py
    parso/python/parser.py
    parso/python/pep8.py
    parso/python/prefix.py
    parso/python/token.py
    parso/python/tokenize.py
    parso/python/tree.py
    parso/tree.py
    parso/utils.py
)

RESOURCE_FILES(
    PREFIX contrib/python/parso/py2/
    .dist-info/METADATA
    .dist-info/top_level.txt
    parso/python/grammar27.txt
    parso/python/grammar310.txt
    parso/python/grammar33.txt
    parso/python/grammar34.txt
    parso/python/grammar35.txt
    parso/python/grammar36.txt
    parso/python/grammar37.txt
    parso/python/grammar38.txt
    parso/python/grammar39.txt
)

NO_LINT()

END()

RECURSE_FOR_TESTS(
    tests
)
