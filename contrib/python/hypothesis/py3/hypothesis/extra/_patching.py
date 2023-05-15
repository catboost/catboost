# This file is part of Hypothesis, which may be found at
# https://github.com/HypothesisWorks/hypothesis/
#
# Copyright the Hypothesis Authors.
# Individual contributors are listed in AUTHORS.rst and the git log.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""
Write patches which add @example() decorators for discovered test cases.

Requires `hypothesis[codemods,ghostwriter]` installed, i.e. black and libcst.

This module is used by Hypothesis' builtin pytest plugin for failing examples
discovered during testing, and by HypoFuzz for _covering_ examples discovered
during fuzzing.
"""

import difflib
import hashlib
import inspect
import re
import sys
from contextlib import suppress
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import libcst as cst
from libcst.codemod import CodemodContext, VisitorBasedCodemodCommand

from hypothesis.configuration import storage_directory
from hypothesis.extra.codemods import _native_parser
from hypothesis.version import __version__

try:
    import black
except ImportError:
    black = None  # type: ignore

HEADER = f"""\
From HEAD Mon Sep 17 00:00:00 2001
From: Hypothesis {__version__} <no-reply@hypothesis.works>
Date: {{when:%a, %d %b %Y %H:%M:%S}}
Subject: [PATCH] {{msg}}

---
"""
_space_only_re = re.compile("^ +$", re.MULTILINE)
_leading_space_re = re.compile("(^[ ]*)(?:[^ \n])", re.MULTILINE)


def dedent(text):
    # Simplified textwrap.dedent, for valid Python source code only
    text = _space_only_re.sub("", text)
    prefix = min(_leading_space_re.findall(text), key=len)
    return re.sub(r"(?m)^" + prefix, "", text), prefix


def indent(text: str, prefix: str) -> str:
    return "".join(prefix + line for line in text.splitlines(keepends=True))


class AddExamplesCodemod(VisitorBasedCodemodCommand):
    DESCRIPTION = "Add explicit examples to failing tests."

    @classmethod
    def refactor(cls, code: str, fn_examples: dict) -> str:
        """Add @example() decorator(s) for failing test(s).

        `code` is the source code of the module where the test functions are defined.
        `fn_examples` is a dict of function name to list-of-failing-examples.
        """
        dedented, prefix = dedent(code)
        with _native_parser():
            mod = cst.parse_module(dedented)
        modded = cls(CodemodContext(), fn_examples, prefix).transform_module(mod).code
        return indent(modded, prefix=prefix)

    def __init__(self, context, fn_examples, prefix="", via="discovered failure"):
        assert fn_examples, "This codemod does nothing without fn_examples."
        super().__init__(context)

        # Codemod the failing examples to Call nodes usable as decorators
        self.via = via
        self.line_length = 88 - len(prefix)  # to match Black's default formatting
        self.fn_examples = {
            k: tuple(self.__call_node_to_example_dec(ex) for ex in nodes)
            for k, nodes in fn_examples.items()
        }

    def __call_node_to_example_dec(self, node):
        node = node.with_changes(
            func=cst.Name("example"),
            args=[a.with_changes(comma=cst.MaybeSentinel.DEFAULT) for a in node.args]
            if black
            else node.args,
        )
        # Note: calling a method on a decorator requires PEP-614, i.e. Python 3.9+,
        # but plumbing two cases through doesn't seem worth the trouble :-/
        via = cst.Call(
            func=cst.Attribute(node, cst.Name("via")),
            args=[cst.Arg(cst.SimpleString(repr(self.via)))],
        )
        if black:  # pragma: no branch
            pretty = black.format_str(
                cst.Module([]).code_for_node(via),
                mode=black.FileMode(line_length=self.line_length),
            )
            via = cst.parse_expression(pretty.strip())
        return cst.Decorator(via)

    def leave_FunctionDef(self, _, updated_node):
        return updated_node.with_changes(
            # TODO: improve logic for where in the list to insert this decorator
            decorators=updated_node.decorators
            + self.fn_examples.get(updated_node.name.value, ())
        )


def get_patch_for(func, failing_examples):
    # Skip this if we're unable to find the location or source of this function.
    try:
        fname = Path(sys.modules[func.__module__].__file__).relative_to(Path.cwd())
        before = inspect.getsource(func)
    except Exception:
        return None

    # The printed examples might include object reprs which are invalid syntax,
    # so we parse here and skip over those.  If _none_ are valid, there's no patch.
    call_nodes = []
    for ex in failing_examples:
        with suppress(Exception):
            node = cst.parse_expression(ex)
            assert isinstance(node, cst.Call), node
            call_nodes.append(node)
    if not call_nodes:
        return None

    # Do the codemod and return a triple containing location and replacement info.
    after = AddExamplesCodemod.refactor(
        before,
        fn_examples={func.__name__: call_nodes},
    )
    return (str(fname), before, after)


def make_patch(triples, *, msg="Hypothesis: add failing examples", when=None):
    """Create a patch for (fname, before, after) triples."""
    assert triples, "attempted to create empty patch"
    when = when or datetime.now(tz=timezone.utc)

    by_fname = {}
    for fname, before, after in triples:
        by_fname.setdefault(Path(fname), []).append((before, after))

    diffs = [HEADER.format(msg=msg, when=when)]
    for fname, changes in sorted(by_fname.items()):
        source_before = source_after = fname.read_text(encoding="utf-8")
        for before, after in changes:
            source_after = source_after.replace(before, after, 1)
        ud = difflib.unified_diff(
            source_before.splitlines(keepends=True),
            source_after.splitlines(keepends=True),
            fromfile=str(fname),
            tofile=str(fname),
        )
        diffs.append("".join(ud))
    return "".join(diffs)


def save_patch(patch: str) -> Path:  # pragma: no cover
    today = date.today().isoformat()
    hash = hashlib.sha1(patch.encode()).hexdigest()[:8]
    fname = Path(storage_directory("patches", f"{today}--{hash}.patch"))
    fname.parent.mkdir(parents=True, exist_ok=True)
    fname.write_text(patch, encoding="utf-8")
    return fname.relative_to(Path.cwd())


def gc_patches():  # pragma: no cover
    cutoff = date.today() - timedelta(days=7)
    for fname in Path(storage_directory("patches")).glob("????-??-??--????????.patch"):
        if date.fromisoformat(fname.stem.split("--")[0]) < cutoff:
            fname.unlink()
