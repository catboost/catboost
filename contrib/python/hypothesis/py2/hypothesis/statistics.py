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

import math

from hypothesis.internal.conjecture.data import Status
from hypothesis.internal.conjecture.engine import MAX_SHRINKS, ExitReason
from hypothesis.utils.dynamicvariables import DynamicVariable

collector = DynamicVariable(None)


class Statistics(object):
    def __init__(self, engine):
        self.passing_examples = len(engine.status_runtimes.get(Status.VALID, ()))
        self.invalid_examples = len(
            engine.status_runtimes.get(Status.INVALID, [])
            + engine.status_runtimes.get(Status.OVERRUN, [])
        )
        self.failing_examples = len(engine.status_runtimes.get(Status.INTERESTING, ()))
        self.targets = dict(engine.best_observed_targets)

        runtimes = sorted(
            engine.status_runtimes.get(Status.VALID, [])
            + engine.status_runtimes.get(Status.INVALID, [])
            + engine.status_runtimes.get(Status.INTERESTING, [])
        )

        self.has_runs = bool(runtimes)
        if not self.has_runs:
            return

        n = max(0, len(runtimes) - 1)
        lower = int(runtimes[int(math.floor(n * 0.05))] * 1000)
        upper = int(runtimes[int(math.ceil(n * 0.95))] * 1000)
        if upper == 0:
            self.runtimes = "< 1ms"
        elif lower == upper:
            self.runtimes = "~ %dms" % (lower,)
        else:
            self.runtimes = "%d-%d ms" % (lower, upper)

        if engine.exit_reason == ExitReason.finished:
            self.exit_reason = "nothing left to do"
        elif engine.exit_reason == ExitReason.flaky:
            self.exit_reason = "test was flaky"
        elif engine.exit_reason == ExitReason.max_shrinks:
            self.exit_reason = "shrunk example %s times" % (MAX_SHRINKS,)
        elif engine.exit_reason == ExitReason.max_iterations:
            self.exit_reason = (
                "settings.max_examples={}, but < 10% of examples satisfied "
                "assumptions"
            ).format(engine.settings.max_examples)
        else:
            self.exit_reason = "settings.%s=%r" % (
                engine.exit_reason.name,
                getattr(engine.settings, engine.exit_reason.name),
            )

        self.events = [
            "%6.2f%%, %s" % (c / engine.call_count * 100, e)
            for e, c in sorted(
                engine.event_call_counts.items(), key=lambda x: (-x[1], x[0])
            )
        ]

        total_runtime = math.fsum(engine.all_runtimes)
        total_drawtime = math.fsum(engine.all_drawtimes)

        if total_drawtime == 0.0 and total_runtime >= 0.0:
            self.draw_time_percentage = "~ 0%"
        elif total_drawtime < 0.0 or total_runtime <= 0.0:
            # This weird condition is possible in two ways:
            # 1.  drawtime and/or runtime are negative, due to clock changes
            #     on Python 2 or old OSs (we use monotonic() where available)
            # 2.  floating-point issues *very rarely* cause math.fsum to be
            #     off by the lowest bit, so drawtime==0 and runtime!=0, eek!
            self.draw_time_percentage = "NaN"
        else:
            draw_time_percentage = 100.0 * min(1, total_drawtime / total_runtime)

            self.draw_time_percentage = "~ %d%%" % (round(draw_time_percentage),)

    def get_description(self):
        """Return a list of lines describing the statistics, to be printed."""
        if not self.has_runs:
            return ["  - Test was never run"]
        lines = [
            "  - %d passing examples, %d failing examples, %d invalid examples"
            % (self.passing_examples, self.failing_examples, self.invalid_examples),
            "  - Typical runtimes: %s" % (self.runtimes,),
            "  - Fraction of time spent in data generation: %s"
            % (self.draw_time_percentage,),
            "  - Stopped because %s" % (self.exit_reason,),
        ]
        if self.targets:
            lines.append("  - Highest target scores:")
            for label, score in sorted(self.targets.items(), key=lambda x: x[::-1]):
                lines.append("{:>20g}  ({})".format(score, repr(label)))
        if self.events:
            lines.append("  - Events:")
            lines += ["    * %s" % (event,) for event in self.events]
        return lines


def note_engine_for_statistics(engine):
    callback = collector.value
    if callback is not None:
        callback(Statistics(engine))
