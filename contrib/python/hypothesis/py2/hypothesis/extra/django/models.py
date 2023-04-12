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

from __future__ import absolute_import, division, print_function

import django.db.models as dm
from django.db import IntegrityError

import hypothesis.strategies._internal.core as st
from hypothesis import reject
from hypothesis._settings import note_deprecation
from hypothesis.errors import InvalidArgument
from hypothesis.extra.django import from_field, register_field_strategy
from hypothesis.utils.conventions import DefaultValueType

if False:
    from typing import Any, Type, List, Text, Union  # noqa


def add_default_field_mapping(field_type, strategy):
    # type: (Type[dm.Field], st.SearchStrategy[Any]) -> None
    note_deprecation(
        "`hypothesis.extra.django.models.add_default_field_mapping` is deprecated; use `hypothesis.extra.django."
        "register_field_strategy` instead.",
        since="2019-01-10",
    )
    register_field_strategy(field_type, strategy)


default_value = DefaultValueType(u"default_value")


@st.defines_strategy
def models(
    model,  # type: Type[dm.Model]
    **field_strategies  # type: Union[st.SearchStrategy[Any], DefaultValueType]
):
    # type: (...) -> st.SearchStrategy[Any]
    """Return a strategy for examples of ``model``.

    .. warning::
        Hypothesis creates saved models. This will run inside your testing
        transaction when using the test runner, but if you use the dev console
        this will leave debris in your database.

    ``model`` must be an subclass of :class:`~django:django.db.models.Model`.
    Strategies for fields may be passed as keyword arguments, for example
    ``is_staff=st.just(False)``.

    Hypothesis can often infer a strategy based the field type and validators
    - for best results, make sure your validators are derived from Django's
    and therefore have the known types and attributes.
    Passing a keyword argument skips inference for that field; pass a strategy
    or pass ``hypothesis.extra.django.models.default_value`` to skip
    inference for that field.

    Foreign keys are not automatically derived. If they're nullable they will
    default to always being null, otherwise you always have to specify them.
    For example, examples of a Shop type with a foreign key to Company could
    be generated with::

      shop_strategy = models(Shop, company=models(Company))
    """
    note_deprecation(
        "`hypothesis.extra.django.models.models` is deprecated; use `hypothesis.extra.django."
        "from_model` instead.",
        since="2019-01-10",
    )
    result = {}
    for k, v in field_strategies.items():
        if not isinstance(v, DefaultValueType):
            result[k] = v
    missed = []  # type: List[Text]
    for f in model._meta.concrete_fields:
        if not (f.name in field_strategies or isinstance(f, dm.AutoField)):
            result[f.name] = from_field(f)
            if result[f.name].is_empty:
                missed.append(f.name)
    if missed:
        raise InvalidArgument(
            u"Missing arguments for mandatory field%s %s for model %s"
            % (u"s" if len(missed) > 1 else u"", u", ".join(missed), model.__name__)
        )

    for field in result:
        if model._meta.get_field(field).primary_key:
            # The primary key is generated as part of the strategy. We
            # want to find any existing row with this primary key and
            # overwrite its contents.
            kwargs = {field: result.pop(field)}
            kwargs["defaults"] = st.fixed_dictionaries(result)
            return _models_impl(st.builds(model.objects.update_or_create, **kwargs))

    # The primary key is not generated as part of the strategy, so we
    # just match against any row that has the same value for all
    # fields.
    return _models_impl(st.builds(model.objects.get_or_create, **result))


@st.composite
def _models_impl(draw, strat):
    """Handle the nasty part of drawing a value for models()"""
    try:
        return draw(strat)[0]
    except IntegrityError:
        reject()
