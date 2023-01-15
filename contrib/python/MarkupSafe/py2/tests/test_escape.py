# -*- coding: utf-8 -*-
import pytest

from markupsafe import Markup


@pytest.mark.parametrize(
    ("value", "expect"),
    (
        # empty
        (u"", u""),
        # ascii
        (u"abcd&><'\"efgh", u"abcd&amp;&gt;&lt;&#39;&#34;efgh"),
        (u"&><'\"efgh", u"&amp;&gt;&lt;&#39;&#34;efgh"),
        (u"abcd&><'\"", u"abcd&amp;&gt;&lt;&#39;&#34;"),
        # 2 byte
        (u"こんにちは&><'\"こんばんは", u"こんにちは&amp;&gt;&lt;&#39;&#34;こんばんは"),
        (u"&><'\"こんばんは", u"&amp;&gt;&lt;&#39;&#34;こんばんは"),
        (u"こんにちは&><'\"", u"こんにちは&amp;&gt;&lt;&#39;&#34;"),
        # 4 byte
        (
            u"\U0001F363\U0001F362&><'\"\U0001F37A xyz",
            u"\U0001F363\U0001F362&amp;&gt;&lt;&#39;&#34;\U0001F37A xyz",
        ),
        (u"&><'\"\U0001F37A xyz", u"&amp;&gt;&lt;&#39;&#34;\U0001F37A xyz"),
        (u"\U0001F363\U0001F362&><'\"", u"\U0001F363\U0001F362&amp;&gt;&lt;&#39;&#34;"),
    ),
)
def test_escape(escape, value, expect):
    assert escape(value) == Markup(expect)
