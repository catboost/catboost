# coding=utf-8
#
# This file is part of Hypothesis, which may be found at
# https://github.com/HypothesisWorks/hypothesis/
#
# Most of this work is copyright (C) 2013-2019 David R. MacIver
# (david@drmaciver.com), but it contains contributions by others. See
# CONTRIBUTING.rst for a full list of people who may hold copyright, and
# consult the git log if you need to determine who owns an individual
# contribution.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.
#
# END HEADER

"""This module contains various provisional APIs and strategies.

It is intended for internal use, to ease code reuse, and is not stable.
Point releases may move or break the contents at any time!

Internet strategies should conform to :rfc:`3986` or the authoritative
definitions it links to.  If not, report the bug!
"""
# https://tools.ietf.org/html/rfc3696

from __future__ import absolute_import, division, print_function

import os.path
import string

import importlib_resources

import hypothesis.internal.conjecture.utils as cu
import hypothesis.strategies._internal.core as st
from hypothesis.errors import InvalidArgument
from hypothesis.strategies._internal.strategies import SearchStrategy

if False:
    from typing import Text  # noqa
    from hypothesis.strategies._internal.strategies import SearchStrategy, Ex  # noqa


URL_SAFE_CHARACTERS = frozenset(string.ascii_letters + string.digits + "$-_.+!*'(),")


# This file is sourced from http://data.iana.org/TLD/tlds-alpha-by-domain.txt
# The file contains additional information about the date that it was last updated.
_tlds = importlib_resources.read_text("hypothesis.vendor", "tlds-alpha-by-domain.txt").splitlines()
assert _tlds[0].startswith("#")
TOP_LEVEL_DOMAINS = ["COM"] + sorted(_tlds[1:], key=len)


class DomainNameStrategy(SearchStrategy):
    @staticmethod
    def clean_inputs(minimum, maximum, value, variable_name):
        if value is None:
            value = maximum
        elif not isinstance(value, int):
            raise InvalidArgument(
                "Expected integer but %s is a %s"
                % (variable_name, type(value).__name__)
            )
        elif not minimum <= value <= maximum:
            raise InvalidArgument(
                "Invalid value %r < %s=%r < %r"
                % (minimum, variable_name, value, maximum)
            )
        return value

    def __init__(self, max_length=None, max_element_length=None):
        """
        A strategy for :rfc:`1035` fully qualified domain names.

        The upper limit for max_length is 255 in accordance with :rfc:`1035#section-2.3.4`
        The lower limit for max_length is 4, corresponding to a two letter domain
        with a single letter subdomain.
        The upper limit for max_element_length is 63 in accordance with :rfc:`1035#section-2.3.4`
        The lower limit for max_element_length is 1 in accordance with :rfc:`1035#section-2.3.4`
        """
        # https://tools.ietf.org/html/rfc1035#section-2.3.4

        max_length = self.clean_inputs(4, 255, max_length, "max_length")
        max_element_length = self.clean_inputs(
            1, 63, max_element_length, "max_element_length"
        )

        super(DomainNameStrategy, self).__init__()
        self.max_length = max_length
        self.max_element_length = max_element_length

        # These regular expressions are constructed to match the documented
        # information in https://tools.ietf.org/html/rfc1035#section-2.3.1
        # which defines the allowed syntax of a subdomain string.
        if self.max_element_length == 1:
            self.label_regex = r"[a-zA-Z]"
        elif self.max_element_length == 2:
            self.label_regex = r"[a-zA-Z][a-zA-Z0-9]?"
        else:
            maximum_center_character_pattern_repetitions = self.max_element_length - 2
            self.label_regex = r"[a-zA-Z]([a-zA-Z0-9\-]{0,%d}[a-zA-Z0-9])?" % (
                maximum_center_character_pattern_repetitions,
            )

    def do_draw(self, data):
        # 1 - Select a valid top-level domain (TLD) name
        # 2 - Check that the number of characters in our selected TLD won't
        # prevent us from generating at least a 1 character subdomain.
        # 3 - Randomize the TLD between upper and lower case characters.
        domain = data.draw(
            st.sampled_from(TOP_LEVEL_DOMAINS)
            .filter(lambda tld: len(tld) + 2 <= self.max_length)
            .flatmap(
                lambda tld: st.tuples(
                    *[st.sampled_from([c.lower(), c.upper()]) for c in tld]
                ).map(u"".join)
            )
        )
        # The maximum possible number of subdomains is 126,
        # 1 character subdomain + 1 '.' character, * 126 = 252,
        # with a max of 255, that leaves 3 characters for a TLD.
        # Allowing any more subdomains would not leave enough
        # characters for even the shortest possible TLDs.
        elements = cu.many(data, min_size=1, average_size=1, max_size=126)
        while elements.more():
            # Generate a new valid subdomain using the regex strategy.
            sub_domain = data.draw(st.from_regex(self.label_regex, fullmatch=True))
            if len(domain) + len(sub_domain) >= self.max_length:
                data.stop_example(discard=True)
                break
            domain = sub_domain + "." + domain
        return domain


@st.defines_strategy_with_reusable_values
def domains(
    max_length=255,  # type: int
    max_element_length=63,  # type: int
):
    # type: (...) -> SearchStrategy[Text]
    """Generate :rfc:`1035` compliant fully qualified domain names."""
    return DomainNameStrategy(
        max_length=max_length, max_element_length=max_element_length
    )


@st.defines_strategy_with_reusable_values
def urls():
    # type: () -> SearchStrategy[Text]
    """A strategy for :rfc:`3986`, generating http/https URLs."""

    def url_encode(s):
        return "".join(c if c in URL_SAFE_CHARACTERS else "%%%02X" % ord(c) for c in s)

    schemes = st.sampled_from(["http", "https"])
    ports = st.integers(min_value=0, max_value=2 ** 16 - 1).map(":{}".format)
    paths = st.lists(st.text(string.printable).map(url_encode)).map("/".join)

    return st.builds(
        u"{}://{}{}/{}".format, schemes, domains(), st.just(u"") | ports, paths
    )


@st.defines_strategy_with_reusable_values
def ip4_addr_strings():
    # type: () -> SearchStrategy[Text]
    """A strategy for IPv4 address strings.

    This consists of four strings representing integers [0..255],
    without zero-padding, joined by dots.
    """
    return st.builds(u"{}.{}.{}.{}".format, *(4 * [st.integers(0, 255)]))


@st.defines_strategy_with_reusable_values
def ip6_addr_strings():
    # type: () -> SearchStrategy[Text]
    """A strategy for IPv6 address strings.

    This consists of sixteen quads of hex digits (0000 .. FFFF), joined
    by colons.  Values do not currently have zero-segments collapsed.
    """
    part = st.integers(0, 2 ** 16 - 1).map(u"{:04x}".format)
    return st.tuples(*[part] * 8).map(lambda a: u":".join(a).upper())
