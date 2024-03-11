# This file is part of Hypothesis, which may be found at
# https://github.com/HypothesisWorks/hypothesis/
#
# Copyright the Hypothesis Authors.
# Individual contributors are listed in AUTHORS.rst and the git log.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

import functools
import os
import subprocess
import sys
import types
from collections import defaultdict
from functools import lru_cache, reduce
from os import sep
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Set, Tuple

from hypothesis._settings import Phase, Verbosity
from hypothesis.internal.escalation import is_hypothesis_file

if TYPE_CHECKING:
    from typing import TypeAlias
else:
    TypeAlias = object

Location: TypeAlias = Tuple[str, int]
Branch: TypeAlias = Tuple[Optional[Location], Location]
Trace: TypeAlias = Set[Branch]


@lru_cache(maxsize=None)
def should_trace_file(fname):
    # fname.startswith("<") indicates runtime code-generation via compile,
    # e.g. compile("def ...", "<string>", "exec") in e.g. attrs methods.
    return not (is_hypothesis_file(fname) or fname.startswith("<"))


# where possible, we'll use 3.12's new sys.monitoring module for low-overhead
# coverage instrumentation; on older python versions we'll use sys.settrace.
# tool_id = 1 is designated for coverage, but we intentionally choose a
# non-reserved tool id so we can co-exist with coverage tools.
MONITORING_TOOL_ID = 3
if sys.version_info[:2] >= (3, 12):
    MONITORING_EVENTS = {sys.monitoring.events.LINE: "trace_line"}


class Tracer:
    """A super-simple branch coverage tracer."""

    __slots__ = ("branches", "_previous_location")

    def __init__(self):
        self.branches: Trace = set()
        self._previous_location = None

    def trace(self, frame, event, arg):
        if event == "call":
            return self.trace
        elif event == "line":
            # manual inlining of self.trace_line for performance.
            fname = frame.f_code.co_filename
            if should_trace_file(fname):
                current_location = (fname, frame.f_lineno)
                self.branches.add((self._previous_location, current_location))
                self._previous_location = current_location

    def trace_line(self, code: types.CodeType, line_number: int) -> None:
        fname = code.co_filename
        if should_trace_file(fname):
            current_location = (fname, line_number)
            self.branches.add((self._previous_location, current_location))
            self._previous_location = current_location

    def __enter__(self):
        if sys.version_info[:2] < (3, 12):
            assert sys.gettrace() is None  # caller checks in core.py
            sys.settrace(self.trace)
            return self

        sys.monitoring.use_tool_id(MONITORING_TOOL_ID, "scrutineer")
        for event, callback_name in MONITORING_EVENTS.items():
            sys.monitoring.set_events(MONITORING_TOOL_ID, event)
            callback = getattr(self, callback_name)
            sys.monitoring.register_callback(MONITORING_TOOL_ID, event, callback)

        return self

    def __exit__(self, *args, **kwargs):
        if sys.version_info[:2] < (3, 12):
            sys.settrace(None)
            return

        sys.monitoring.free_tool_id(MONITORING_TOOL_ID)
        for event in MONITORING_EVENTS:
            sys.monitoring.register_callback(MONITORING_TOOL_ID, event, None)


UNHELPFUL_LOCATIONS = (
    # There's a branch which is only taken when an exception is active while exiting
    # a contextmanager; this is probably after the fault has been triggered.
    # Similar reasoning applies to a few other standard-library modules: even
    # if the fault was later, these still aren't useful locations to report!
    f"{sep}contextlib.py",
    f"{sep}inspect.py",
    f"{sep}re.py",
    f"{sep}re{sep}__init__.py",  # refactored in Python 3.11
    f"{sep}warnings.py",
    # Quite rarely, the first AFNP line is in Pytest's internals.
    f"{sep}_pytest{sep}assertion{sep}__init__.py",
    f"{sep}_pytest{sep}assertion{sep}rewrite.py",
    f"{sep}_pytest{sep}_io{sep}saferepr.py",
    f"{sep}pluggy{sep}_result.py",
)


def get_explaining_locations(traces):
    # Traces is a dict[interesting_origin | None, set[frozenset[tuple[str, int]]]]
    # Each trace in the set might later become a Counter instead of frozenset.
    if not traces:
        return {}

    unions = {origin: set().union(*values) for origin, values in traces.items()}
    seen_passing = {None}.union(*unions.pop(None, set()))

    always_failing_never_passing = {
        origin: reduce(set.intersection, [set().union(*v) for v in values])
        - seen_passing
        for origin, values in traces.items()
        if origin is not None
    }

    # Build the observed parts of the control-flow graph for each origin
    cf_graphs = {origin: defaultdict(set) for origin in unions}
    for origin, seen_arcs in unions.items():
        for src, dst in seen_arcs:
            cf_graphs[origin][src].add(dst)
        assert cf_graphs[origin][None], "Expected start node with >=1 successor"

    # For each origin, our explanation is the always_failing_never_passing lines
    # which are reachable from the start node (None) without passing through another
    # AFNP line.  So here's a whatever-first search with early stopping:
    explanations = defaultdict(set)
    for origin in unions:
        queue = {None}
        seen = set()
        while queue:
            assert queue.isdisjoint(seen), f"Intersection: {queue & seen}"
            src = queue.pop()
            seen.add(src)
            if src in always_failing_never_passing[origin]:
                explanations[origin].add(src)
            else:
                queue.update(cf_graphs[origin][src] - seen)

    # The last step is to filter out explanations that we know would be uninformative.
    # When this is the first AFNP location, we conclude that Scrutineer missed the
    # real divergence (earlier in the trace) and drop that unhelpful explanation.
    return {
        origin: {loc for loc in afnp_locs if not loc[0].endswith(UNHELPFUL_LOCATIONS)}
        for origin, afnp_locs in explanations.items()
    }


LIB_DIR = str(Path(sys.executable).parent / "lib")
EXPLANATION_STUB = (
    "Explanation:",
    "    These lines were always and only run by failing examples:",
)


def make_report(explanations, cap_lines_at=5):
    report = defaultdict(list)
    for origin, locations in explanations.items():
        report_lines = [f"        {fname}:{lineno}" for fname, lineno in locations]
        report_lines.sort(key=lambda line: (line.startswith(LIB_DIR), line))
        if len(report_lines) > cap_lines_at + 1:
            msg = "        (and {} more with settings.verbosity >= verbose)"
            report_lines[cap_lines_at:] = [msg.format(len(report_lines[cap_lines_at:]))]
        if report_lines:  # We might have filtered out every location as uninformative.
            report[origin] = list(EXPLANATION_STUB) + report_lines
    return report


def explanatory_lines(traces, settings):
    if Phase.explain in settings.phases and sys.gettrace() and not traces:
        return defaultdict(list)
    # Return human-readable report lines summarising the traces
    explanations = get_explaining_locations(traces)
    max_lines = 5 if settings.verbosity <= Verbosity.normal else float("inf")
    return make_report(explanations, cap_lines_at=max_lines)


# beware the code below; we're using some heuristics to make a nicer report...


@functools.lru_cache
def _get_git_repo_root() -> Path:
    try:
        where = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            check=True,
            timeout=10,
            capture_output=True,
            text=True,
            encoding="utf-8",
        ).stdout.strip()
    except Exception:  # pragma: no cover
        return Path().absolute().parents[-1]
    else:
        return Path(where)


if sys.version_info[:2] <= (3, 8):

    def is_relative_to(self, other):
        return other == self or other in self.parents

else:
    is_relative_to = Path.is_relative_to


def tractable_coverage_report(trace: Trace) -> Dict[str, List[int]]:
    """Report a simple coverage map which is (probably most) of the user's code."""
    coverage: dict = {}
    t = dict(trace)
    for file, line in set(t.keys()).union(t.values()) - {None}:  # type: ignore
        # On Python <= 3.11, we can use coverage.py xor Hypothesis' tracer,
        # so the trace will be empty and this line never run under coverage.
        coverage.setdefault(file, set()).add(line)  # pragma: no cover
    stdlib_fragment = f"{os.sep}lib{os.sep}python3.{sys.version_info.minor}{os.sep}"
    return {
        k: sorted(v)
        for k, v in coverage.items()
        if stdlib_fragment not in k
        and is_relative_to(p := Path(k), _get_git_repo_root())
        and "site-packages" not in p.parts
    }
