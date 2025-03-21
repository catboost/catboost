.. image:: https://img.shields.io/pypi/v/jaraco.collections.svg
   :target: https://pypi.org/project/jaraco.collections

.. image:: https://img.shields.io/pypi/pyversions/jaraco.collections.svg

.. image:: https://github.com/jaraco/jaraco.collections/actions/workflows/main.yml/badge.svg
   :target: https://github.com/jaraco/jaraco.collections/actions?query=workflow%3A%22tests%22
   :alt: tests

.. image:: https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/charliermarsh/ruff/main/assets/badge/v2.json
    :target: https://github.com/astral-sh/ruff
    :alt: Ruff

.. image:: https://readthedocs.org/projects/jaracocollections/badge/?version=latest
   :target: https://jaracocollections.readthedocs.io/en/latest/?badge=latest

.. image:: https://img.shields.io/badge/skeleton-2024-informational
   :target: https://blog.jaraco.com/skeleton

.. image:: https://tidelift.com/badges/package/pypi/jaraco.collections
   :target: https://tidelift.com/subscription/pkg/pypi-jaraco.collections?utm_source=pypi-jaraco.collections&utm_medium=readme

Models and classes to supplement the stdlib 'collections' module.

See the docs, linked above, for descriptions and usage examples.

Highlights include:

- RangeMap: A mapping that accepts a range of values for keys.
- Projection: A subset over an existing mapping.
- KeyTransformingDict: Generalized mapping with keys transformed by a function.
- FoldedCaseKeyedDict: A dict whose string keys are case-insensitive.
- BijectiveMap: A map where keys map to values and values back to their keys.
- ItemsAsAttributes: A mapping mix-in exposing items as attributes.
- IdentityOverrideMap: A map whose keys map by default to themselves unless overridden.
- FrozenDict: A hashable, immutable map.
- Enumeration: An object whose keys are enumerated.
- Everything: A container that contains all things.
- Least, Greatest: Objects that are always less than or greater than any other.
- pop_all: Return all items from the mutable sequence and remove them from that sequence.
- DictStack: A stack of dicts, great for sharing scopes.
- WeightedLookup: A specialized RangeMap for selecting an item by weights.

For Enterprise
==============

Available as part of the Tidelift Subscription.

This project and the maintainers of thousands of other packages are working with Tidelift to deliver one enterprise subscription that covers all of the open source you use.

`Learn more <https://tidelift.com/subscription/pkg/pypi-jaraco.collections?utm_source=pypi-jaraco.collections&utm_medium=referral&utm_campaign=github>`_.
