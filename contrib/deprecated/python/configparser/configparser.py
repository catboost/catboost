#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Convenience module importing everything from backports.configparser."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from backports.configparser import (
    RawConfigParser,
    ConfigParser,
    SafeConfigParser,
    SectionProxy,
    Interpolation,
    BasicInterpolation,
    ExtendedInterpolation,
    LegacyInterpolation,
    NoSectionError,
    DuplicateSectionError,
    DuplicateOptionError,
    NoOptionError,
    InterpolationError,
    InterpolationMissingOptionError,
    InterpolationSyntaxError,
    InterpolationDepthError,
    ParsingError,
    MissingSectionHeaderError,
    ConverterMapping,
    DEFAULTSECT,
    MAX_INTERPOLATION_DEPTH,
)

from backports.configparser import Error, _UNSET, _default_dict, _ChainMap  # noqa: F401

__all__ = [
    "NoSectionError",
    "DuplicateOptionError",
    "DuplicateSectionError",
    "NoOptionError",
    "InterpolationError",
    "InterpolationDepthError",
    "InterpolationMissingOptionError",
    "InterpolationSyntaxError",
    "ParsingError",
    "MissingSectionHeaderError",
    "ConfigParser",
    "SafeConfigParser",
    "RawConfigParser",
    "Interpolation",
    "BasicInterpolation",
    "ExtendedInterpolation",
    "LegacyInterpolation",
    "SectionProxy",
    "ConverterMapping",
    "DEFAULTSECT",
    "MAX_INTERPOLATION_DEPTH",
]

# NOTE: names missing from __all__ imported anyway for backwards compatibility.
