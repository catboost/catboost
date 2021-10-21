# -*- coding: utf-8 -*-
import pytest

from jinja2 import DictLoader
from jinja2 import Environment
from jinja2.exceptions import TemplateNotFound
from jinja2.exceptions import TemplatesNotFound
from jinja2.exceptions import TemplateSyntaxError


@pytest.fixture
def test_env():
    env = Environment(
        loader=DictLoader(
            dict(
                module="{% macro test() %}[{{ foo }}|{{ bar }}]{% endmacro %}",
                header="[{{ foo }}|{{ 23 }}]",
                o_printer="({{ o }})",
            )
        )
    )
    env.globals["bar"] = 23
    return env


class TestImports(object):
    def test_context_imports(self, test_env):
        t = test_env.from_string('{% import "module" as m %}{{ m.test() }}')
        assert t.render(foo=42) == "[|23]"
        t = test_env.from_string(
            '{% import "module" as m without context %}{{ m.test() }}'
        )
        assert t.render(foo=42) == "[|23]"
        t = test_env.from_string(
            '{% import "module" as m with context %}{{ m.test() }}'
        )
        assert t.render(foo=42) == "[42|23]"
        t = test_env.from_string('{% from "module" import test %}{{ test() }}')
        assert t.render(foo=42) == "[|23]"
        t = test_env.from_string(
            '{% from "module" import test without context %}{{ test() }}'
        )
        assert t.render(foo=42) == "[|23]"
        t = test_env.from_string(
            '{% from "module" import test with context %}{{ test() }}'
        )
        assert t.render(foo=42) == "[42|23]"

    def test_import_needs_name(self, test_env):
        test_env.from_string('{% from "foo" import bar %}')
        test_env.from_string('{% from "foo" import bar, baz %}')

        with pytest.raises(TemplateSyntaxError):
            test_env.from_string('{% from "foo" import %}')

    def test_no_trailing_comma(self, test_env):
        with pytest.raises(TemplateSyntaxError):
            test_env.from_string('{% from "foo" import bar, %}')

        with pytest.raises(TemplateSyntaxError):
            test_env.from_string('{% from "foo" import bar,, %}')

        with pytest.raises(TemplateSyntaxError):
            test_env.from_string('{% from "foo" import, %}')

    def test_trailing_comma_with_context(self, test_env):
        test_env.from_string('{% from "foo" import bar, baz with context %}')
        test_env.from_string('{% from "foo" import bar, baz, with context %}')
        test_env.from_string('{% from "foo" import bar, with context %}')
        test_env.from_string('{% from "foo" import bar, with, context %}')
        test_env.from_string('{% from "foo" import bar, with with context %}')

        with pytest.raises(TemplateSyntaxError):
            test_env.from_string('{% from "foo" import bar,, with context %}')

        with pytest.raises(TemplateSyntaxError):
            test_env.from_string('{% from "foo" import bar with context, %}')

    def test_exports(self, test_env):
        m = test_env.from_string(
            """
            {% macro toplevel() %}...{% endmacro %}
            {% macro __private() %}...{% endmacro %}
            {% set variable = 42 %}
            {% for item in [1] %}
                {% macro notthere() %}{% endmacro %}
            {% endfor %}
        """
        ).module
        assert m.toplevel() == "..."
        assert not hasattr(m, "__missing")
        assert m.variable == 42
        assert not hasattr(m, "notthere")


class TestIncludes(object):
    def test_context_include(self, test_env):
        t = test_env.from_string('{% include "header" %}')
        assert t.render(foo=42) == "[42|23]"
        t = test_env.from_string('{% include "header" with context %}')
        assert t.render(foo=42) == "[42|23]"
        t = test_env.from_string('{% include "header" without context %}')
        assert t.render(foo=42) == "[|23]"

    def test_choice_includes(self, test_env):
        t = test_env.from_string('{% include ["missing", "header"] %}')
        assert t.render(foo=42) == "[42|23]"

        t = test_env.from_string('{% include ["missing", "missing2"] ignore missing %}')
        assert t.render(foo=42) == ""

        t = test_env.from_string('{% include ["missing", "missing2"] %}')
        pytest.raises(TemplateNotFound, t.render)
        with pytest.raises(TemplatesNotFound) as e:
            t.render()

        assert e.value.templates == ["missing", "missing2"]
        assert e.value.name == "missing2"

        def test_includes(t, **ctx):
            ctx["foo"] = 42
            assert t.render(ctx) == "[42|23]"

        t = test_env.from_string('{% include ["missing", "header"] %}')
        test_includes(t)
        t = test_env.from_string("{% include x %}")
        test_includes(t, x=["missing", "header"])
        t = test_env.from_string('{% include [x, "header"] %}')
        test_includes(t, x="missing")
        t = test_env.from_string("{% include x %}")
        test_includes(t, x="header")
        t = test_env.from_string("{% include [x] %}")
        test_includes(t, x="header")

    def test_include_ignoring_missing(self, test_env):
        t = test_env.from_string('{% include "missing" %}')
        pytest.raises(TemplateNotFound, t.render)
        for extra in "", "with context", "without context":
            t = test_env.from_string(
                '{% include "missing" ignore missing ' + extra + " %}"
            )
            assert t.render() == ""

    def test_context_include_with_overrides(self, test_env):
        env = Environment(
            loader=DictLoader(
                dict(
                    main="{% for item in [1, 2, 3] %}{% include 'item' %}{% endfor %}",
                    item="{{ item }}",
                )
            )
        )
        assert env.get_template("main").render() == "123"

    def test_unoptimized_scopes(self, test_env):
        t = test_env.from_string(
            """
            {% macro outer(o) %}
            {% macro inner() %}
            {% include "o_printer" %}
            {% endmacro %}
            {{ inner() }}
            {% endmacro %}
            {{ outer("FOO") }}
        """
        )
        assert t.render().strip() == "(FOO)"

    def test_import_from_with_context(self):
        env = Environment(
            loader=DictLoader({"a": "{% macro x() %}{{ foobar }}{% endmacro %}"})
        )
        t = env.from_string(
            "{% set foobar = 42 %}{% from 'a' import x with context %}{{ x() }}"
        )
        assert t.render() == "42"
