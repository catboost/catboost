"""Tests for traitlets.traitlets."""

# Copyright (c) IPython Development Team.
# Distributed under the terms of the Modified BSD License.
#
from __future__ import annotations

from traitlets import Dict, Instance, Integer, Unicode, Union
from traitlets.config import Configurable


def test_handle_docstring():
    class SampleConfigurable(Configurable):
        pass

    class TraitTypesSampleConfigurable(Configurable):
        """TraitTypesSampleConfigurable docstring"""

        trait_integer = Integer(
            help="""trait_integer help text""",
            config=True,
        )
        trait_integer_nohelp = Integer(
            config=True,
        )
        trait_integer_noconfig = Integer(
            help="""trait_integer_noconfig help text""",
        )

        trait_unicode = Unicode(
            help="""trait_unicode help text""",
            config=True,
        )
        trait_unicode_nohelp = Unicode(
            config=True,
        )
        trait_unicode_noconfig = Unicode(
            help="""trait_unicode_noconfig help text""",
        )

        trait_dict = Dict(
            help="""trait_dict help text""",
            config=True,
        )
        trait_dict_nohelp = Dict(
            config=True,
        )
        trait_dict_noconfig = Dict(
            help="""trait_dict_noconfig help text""",
        )

        trait_instance = Instance(
            klass=SampleConfigurable,
            help="""trait_instance help text""",
            config=True,
        )
        trait_instance_nohelp = Instance(
            klass=SampleConfigurable,
            config=True,
        )
        trait_instance_noconfig = Instance(
            klass=SampleConfigurable,
            help="""trait_instance_noconfig help text""",
        )

        trait_union = Union(
            [Integer(), Unicode()],
            help="""trait_union help text""",
            config=True,
        )
        trait_union_nohelp = Union(
            [Integer(), Unicode()],
            config=True,
        )
        trait_union_noconfig = Union(
            [Integer(), Unicode()],
            help="""trait_union_noconfig help text""",
        )

    base_names = SampleConfigurable().trait_names()
    for name in TraitTypesSampleConfigurable().trait_names():
        if name in base_names:
            continue
        doc = getattr(TraitTypesSampleConfigurable, name).__doc__
        if "nohelp" not in name:
            assert doc == f"{name} help text"
