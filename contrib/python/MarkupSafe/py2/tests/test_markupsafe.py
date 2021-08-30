# -*- coding: utf-8 -*-
import pytest

from markupsafe import escape
from markupsafe import escape_silent
from markupsafe import Markup
from markupsafe._compat import PY2
from markupsafe._compat import text_type


def test_adding():
    unsafe = '<script type="application/x-some-script">alert("foo");</script>'
    safe = Markup("<em>username</em>")
    assert unsafe + safe == text_type(escape(unsafe)) + text_type(safe)


@pytest.mark.parametrize(
    ("template", "data", "expect"),
    (
        ("<em>%s</em>", "<bad user>", "<em>&lt;bad user&gt;</em>"),
        (
            "<em>%(username)s</em>",
            {"username": "<bad user>"},
            "<em>&lt;bad user&gt;</em>",
        ),
        ("%i", 3.14, "3"),
        ("%.2f", 3.14, "3.14"),
    ),
)
def test_string_interpolation(template, data, expect):
    assert Markup(template) % data == expect


def test_type_behavior():
    assert type(Markup("foo") + "bar") is Markup
    x = Markup("foo")
    assert x.__html__() is x


def test_html_interop():
    class Foo(object):
        def __html__(self):
            return "<em>awesome</em>"

        def __unicode__(self):
            return "awesome"

        __str__ = __unicode__

    assert Markup(Foo()) == "<em>awesome</em>"
    result = Markup("<strong>%s</strong>") % Foo()
    assert result == "<strong><em>awesome</em></strong>"


def test_tuple_interpol():
    result = Markup("<em>%s:%s</em>") % ("<foo>", "<bar>")
    expect = Markup(u"<em>&lt;foo&gt;:&lt;bar&gt;</em>")
    assert result == expect


def test_dict_interpol():
    result = Markup("<em>%(foo)s</em>") % {"foo": "<foo>"}
    expect = Markup(u"<em>&lt;foo&gt;</em>")
    assert result == expect

    result = Markup("<em>%(foo)s:%(bar)s</em>") % {"foo": "<foo>", "bar": "<bar>"}
    expect = Markup(u"<em>&lt;foo&gt;:&lt;bar&gt;</em>")
    assert result == expect


def test_escaping():
    assert escape("\"<>&'") == "&#34;&lt;&gt;&amp;&#39;"
    assert Markup("<em>Foo &amp; Bar</em>").striptags() == "Foo & Bar"


def test_unescape():
    assert Markup("&lt;test&gt;").unescape() == "<test>"

    result = Markup("jack & tavi are cooler than mike &amp; russ").unescape()
    expect = "jack & tavi are cooler than mike & russ"
    assert result == expect

    original = "&foo&#x3b;"
    once = Markup(original).unescape()
    twice = Markup(once).unescape()
    expect = "&foo;"
    assert once == expect
    assert twice == expect


def test_format():
    result = Markup("<em>{awesome}</em>").format(awesome="<awesome>")
    assert result == "<em>&lt;awesome&gt;</em>"

    result = Markup("{0[1][bar]}").format([0, {"bar": "<bar/>"}])
    assert result == "&lt;bar/&gt;"

    result = Markup("{0[1][bar]}").format([0, {"bar": Markup("<bar/>")}])
    assert result == "<bar/>"


def test_formatting_empty():
    formatted = Markup("{}").format(0)
    assert formatted == Markup("0")


def test_custom_formatting():
    class HasHTMLOnly(object):
        def __html__(self):
            return Markup("<foo>")

    class HasHTMLAndFormat(object):
        def __html__(self):
            return Markup("<foo>")

        def __html_format__(self, spec):
            return Markup("<FORMAT>")

    assert Markup("{0}").format(HasHTMLOnly()) == Markup("<foo>")
    assert Markup("{0}").format(HasHTMLAndFormat()) == Markup("<FORMAT>")


def test_complex_custom_formatting():
    class User(object):
        def __init__(self, id, username):
            self.id = id
            self.username = username

        def __html_format__(self, format_spec):
            if format_spec == "link":
                return Markup('<a href="/user/{0}">{1}</a>').format(
                    self.id, self.__html__()
                )
            elif format_spec:
                raise ValueError("Invalid format spec")

            return self.__html__()

        def __html__(self):
            return Markup("<span class=user>{0}</span>").format(self.username)

    user = User(1, "foo")
    result = Markup("<p>User: {0:link}").format(user)
    expect = Markup('<p>User: <a href="/user/1"><span class=user>foo</span></a>')
    assert result == expect


def test_formatting_with_objects():
    class Stringable(object):
        def __unicode__(self):
            return u"строка"

        if PY2:

            def __str__(self):
                return "some other value"

        else:
            __str__ = __unicode__

    assert Markup("{s}").format(s=Stringable()) == Markup(u"строка")


def test_all_set():
    import markupsafe as markup

    for item in markup.__all__:
        getattr(markup, item)


def test_escape_silent():
    assert escape_silent(None) == Markup()
    assert escape(None) == Markup(None)
    assert escape_silent("<foo>") == Markup(u"&lt;foo&gt;")


def test_splitting():
    expect = [Markup("a"), Markup("b")]
    assert Markup("a b").split() == expect
    assert Markup("a b").rsplit() == expect
    assert Markup("a\nb").splitlines() == expect


def test_mul():
    assert Markup("a") * 3 == Markup("aaa")


def test_escape_return_type():
    assert isinstance(escape("a"), Markup)
    assert isinstance(escape(Markup("a")), Markup)

    class Foo:
        def __html__(self):
            return "<strong>Foo</strong>"

    assert isinstance(escape(Foo()), Markup)
