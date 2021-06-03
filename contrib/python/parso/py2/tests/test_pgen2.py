"""Test suite for 2to3's parser and grammar files.

This is the place to add tests for changes to 2to3's grammar, such as those
merging the grammars for Python 2 and 3. In addition to specific tests for
parts of the grammar we've changed, we also make sure we can parse the
test_grammar.py files from both Python 2 and Python 3.
"""

from textwrap import dedent

import pytest

from parso import load_grammar
from parso import ParserSyntaxError
from parso.pgen2 import generate_grammar
from parso.python import tokenize


def _parse(code, version=None):
    code = dedent(code) + "\n\n"
    grammar = load_grammar(version=version)
    return grammar.parse(code, error_recovery=False)


def _invalid_syntax(code, version=None, **kwargs):
    with pytest.raises(ParserSyntaxError):
        module = _parse(code, version=version, **kwargs)
        # For debugging
        print(module.children)


def test_formfeed(each_version):
    s = u"foo\n\x0c\nfoo\n"
    t = _parse(s, each_version)
    assert t.children[0].children[0].type == 'name'
    assert t.children[1].children[0].type == 'name'
    s = u"1\n\x0c\x0c\n2\n"
    t = _parse(s, each_version)

    with pytest.raises(ParserSyntaxError):
        s = u"\n\x0c2\n"
        _parse(s, each_version)


def test_matrix_multiplication_operator(works_ge_py35):
    works_ge_py35.parse("a @ b")
    works_ge_py35.parse("a @= b")


def test_yield_from(works_ge_py3, each_version):
    works_ge_py3.parse("yield from x")
    works_ge_py3.parse("(yield from x) + y")
    _invalid_syntax("yield from", each_version)


def test_await_expr(works_ge_py35):
    works_ge_py35.parse("""async def foo():
                         await x
                  """)

    works_ge_py35.parse("""async def foo():

        def foo(): pass

        def foo(): pass

        await x
    """)

    works_ge_py35.parse("""async def foo(): return await a""")

    works_ge_py35.parse("""def foo():
        def foo(): pass
        async def foo(): await x
    """)


@pytest.mark.skipif('sys.version_info[:2] < (3, 5)')
@pytest.mark.xfail(reason="acting like python 3.7")
def test_async_var():
    _parse("""async = 1""", "3.5")
    _parse("""await = 1""", "3.5")
    _parse("""def async(): pass""", "3.5")


def test_async_for(works_ge_py35):
    works_ge_py35.parse("async def foo():\n async for a in b: pass")


@pytest.mark.parametrize("body", [
    """[1 async for a in b
    ]""",
    """[1 async
    for a in b
    ]""",
    """[
    1
    async for a in b
    ]""",
    """[
    1
    async for a
    in b
    ]""",
    """[
    1
    async
    for
    a
    in
    b
    ]""",
    """  [
    1 async for a in b
  ]""",
])
def test_async_for_comprehension_newline(works_ge_py36, body):
    # Issue #139
    works_ge_py36.parse("""async def foo():
    {}""".format(body))


def test_async_with(works_ge_py35):
    works_ge_py35.parse("async def foo():\n async with a: pass")

    @pytest.mark.skipif('sys.version_info[:2] < (3, 5)')
    @pytest.mark.xfail(reason="acting like python 3.7")
    def test_async_with_invalid():
        _invalid_syntax("""def foo():
                                   async with a: pass""", version="3.5")


def test_raise_3x_style_1(each_version):
    _parse("raise", each_version)


def test_raise_2x_style_2(works_in_py2):
    works_in_py2.parse("raise E, V")

def test_raise_2x_style_3(works_in_py2):
    works_in_py2.parse("raise E, V, T")

def test_raise_2x_style_invalid_1(each_version):
    _invalid_syntax("raise E, V, T, Z", version=each_version)

def test_raise_3x_style(works_ge_py3):
    works_ge_py3.parse("raise E1 from E2")

def test_raise_3x_style_invalid_1(each_version):
    _invalid_syntax("raise E, V from E1", each_version)

def test_raise_3x_style_invalid_2(each_version):
    _invalid_syntax("raise E from E1, E2", each_version)

def test_raise_3x_style_invalid_3(each_version):
    _invalid_syntax("raise from E1, E2", each_version)

def test_raise_3x_style_invalid_4(each_version):
    _invalid_syntax("raise E from", each_version)


# Adapted from Python 3's Lib/test/test_grammar.py:GrammarTests.testFuncdef
def test_annotation_1(works_ge_py3):
    works_ge_py3.parse("""def f(x) -> list: pass""")

def test_annotation_2(works_ge_py3):
    works_ge_py3.parse("""def f(x:int): pass""")

def test_annotation_3(works_ge_py3):
    works_ge_py3.parse("""def f(*x:str): pass""")

def test_annotation_4(works_ge_py3):
    works_ge_py3.parse("""def f(**x:float): pass""")

def test_annotation_5(works_ge_py3):
    works_ge_py3.parse("""def f(x, y:1+2): pass""")

def test_annotation_6(each_py3_version):
    _invalid_syntax("""def f(a, (b:1, c:2, d)): pass""", each_py3_version)

def test_annotation_7(each_py3_version):
    _invalid_syntax("""def f(a, (b:1, c:2, d), e:3=4, f=5, *g:6): pass""", each_py3_version)

def test_annotation_8(each_py3_version):
    s = """def f(a, (b:1, c:2, d), e:3=4, f=5,
                    *g:6, h:7, i=8, j:9=10, **k:11) -> 12: pass"""
    _invalid_syntax(s, each_py3_version)


def test_except_new(each_version):
    s = dedent("""
        try:
            x
        except E as N:
            y""")
    _parse(s, each_version)

def test_except_old(works_in_py2):
    s = dedent("""
        try:
            x
        except E, N:
            y""")
    works_in_py2.parse(s)


# Adapted from Python 3's Lib/test/test_grammar.py:GrammarTests.testAtoms
def test_set_literal_1(works_ge_py27):
    works_ge_py27.parse("""x = {'one'}""")

def test_set_literal_2(works_ge_py27):
    works_ge_py27.parse("""x = {'one', 1,}""")

def test_set_literal_3(works_ge_py27):
    works_ge_py27.parse("""x = {'one', 'two', 'three'}""")

def test_set_literal_4(works_ge_py27):
    works_ge_py27.parse("""x = {2, 3, 4,}""")


def test_new_octal_notation(each_version):
    _parse("""0o7777777777777""", each_version)
    _invalid_syntax("""0o7324528887""", each_version)


def test_old_octal_notation(works_in_py2):
    works_in_py2.parse("07")


def test_long_notation(works_in_py2):
    works_in_py2.parse("0xFl")
    works_in_py2.parse("0xFL")
    works_in_py2.parse("0b1l")
    works_in_py2.parse("0B1L")
    works_in_py2.parse("0o7l")
    works_in_py2.parse("0O7L")
    works_in_py2.parse("0l")
    works_in_py2.parse("0L")
    works_in_py2.parse("10l")
    works_in_py2.parse("10L")


def test_new_binary_notation(each_version):
    _parse("""0b101010""", each_version)
    _invalid_syntax("""0b0101021""", each_version)


def test_class_new_syntax(works_ge_py3):
    works_ge_py3.parse("class B(t=7): pass")
    works_ge_py3.parse("class B(t, *args): pass")
    works_ge_py3.parse("class B(t, **kwargs): pass")
    works_ge_py3.parse("class B(t, *args, **kwargs): pass")
    works_ge_py3.parse("class B(t, y=9, *args, **kwargs): pass")


def test_parser_idempotency_extended_unpacking(works_ge_py3):
    """A cut-down version of pytree_idempotency.py."""
    works_ge_py3.parse("a, *b, c = x\n")
    works_ge_py3.parse("[*a, b] = x\n")
    works_ge_py3.parse("(z, *y, w) = m\n")
    works_ge_py3.parse("for *z, m in d: pass\n")


def test_multiline_bytes_literals(each_version):
    """
    It's not possible to get the same result when using \xaa in Python 2/3,
    because it's treated differently.
    """
    s = u"""
        md5test(b"\xaa" * 80,
                (b"Test Using Larger Than Block-Size Key "
                 b"and Larger Than One Block-Size Data"),
                "6f630fad67cda0ee1fb1f562db3aa53e")
        """
    _parse(s, each_version)


def test_multiline_bytes_tripquote_literals(each_version):
    s = '''
        b"""
        <?xml version="1.0" encoding="UTF-8"?>
        <!DOCTYPE plist PUBLIC "-//Apple Computer//DTD PLIST 1.0//EN">
        """
        '''
    _parse(s, each_version)


def test_ellipsis(works_ge_py3, each_version):
    works_ge_py3.parse("...")
    _parse("[0][...]", version=each_version)


def test_dict_unpacking(works_ge_py35):
    works_ge_py35.parse("{**dict(a=3), foo:2}")


def test_multiline_str_literals(each_version):
    s = u"""
        md5test("\xaa" * 80,
                ("Test Using Larger Than Block-Size Key "
                 "and Larger Than One Block-Size Data"),
                "6f630fad67cda0ee1fb1f562db3aa53e")
        """
    _parse(s, each_version)


def test_py2_backticks(works_in_py2):
    works_in_py2.parse("`1`")


def test_py2_string_prefixes(works_in_py2):
    works_in_py2.parse("ur'1'")
    works_in_py2.parse("Ur'1'")
    works_in_py2.parse("UR'1'")
    _invalid_syntax("ru'1'", works_in_py2.version)


def py_br(each_version):
    _parse('br""', each_version)


def test_py3_rb(works_ge_py3):
    works_ge_py3.parse("rb'1'")
    works_ge_py3.parse("RB'1'")


def test_left_recursion():
    with pytest.raises(ValueError, match='left recursion'):
        generate_grammar('foo: foo NAME\n', tokenize.PythonTokenTypes)


@pytest.mark.parametrize(
    'grammar, error_match', [
        ['foo: bar | baz\nbar: NAME\nbaz: NAME\n',
         r"foo is ambiguous.*given a TokenType\(NAME\).*bar or baz"],
        ['''foo: bar | baz\nbar: 'x'\nbaz: "x"\n''',
         r"foo is ambiguous.*given a ReservedString\(x\).*bar or baz"],
        ['''foo: bar | 'x'\nbar: 'x'\n''',
         r"foo is ambiguous.*given a ReservedString\(x\).*bar or foo"],
        # An ambiguity with the second (not the first) child of a production
        ['outer: "a" [inner] "b" "c"\ninner: "b" "c" [inner]\n',
         r"outer is ambiguous.*given a ReservedString\(b\).*inner or outer"],
        # An ambiguity hidden by a level of indirection (middle)
        ['outer: "a" [middle] "b" "c"\nmiddle: inner\ninner: "b" "c" [inner]\n',
         r"outer is ambiguous.*given a ReservedString\(b\).*middle or outer"],
    ]
)
def test_ambiguities(grammar, error_match):
    with pytest.raises(ValueError, match=error_match):
        generate_grammar(grammar, tokenize.PythonTokenTypes)
