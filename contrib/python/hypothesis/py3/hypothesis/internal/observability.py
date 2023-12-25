# This file is part of Hypothesis, which may be found at
# https://github.com/HypothesisWorks/hypothesis/
#
# Copyright the Hypothesis Authors.
# Individual contributors are listed in AUTHORS.rst and the git log.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Observability tools to spit out analysis-ready tables, one row per test case."""

import json
import os
from datetime import date, timedelta
from typing import Callable, Dict, List, Optional

from hypothesis.configuration import storage_directory
from hypothesis.internal.conjecture.data import ConjectureData, Status

TESTCASE_CALLBACKS: List[Callable[[dict], None]] = []


def deliver_json_blob(value: dict) -> None:
    for callback in TESTCASE_CALLBACKS:
        callback(value)


def make_testcase(
    *,
    start_timestamp: float,
    test_name_or_nodeid: str,
    data: ConjectureData,
    how_generated: str = "unknown",
    string_repr: str = "<unknown>",
    arguments: Optional[dict] = None,
    metadata: Optional[dict] = None,
    coverage: Optional[Dict[str, List[int]]] = None,
) -> dict:
    if data.interesting_origin:
        status_reason = str(data.interesting_origin)
    else:
        status_reason = str(data.events.pop("invalid because", ""))

    return {
        "type": "test_case",
        "run_start": start_timestamp,
        "property": test_name_or_nodeid,
        "status": {
            Status.OVERRUN: "gave_up",
            Status.INVALID: "gave_up",
            Status.VALID: "passed",
            Status.INTERESTING: "failed",
        }[data.status],
        "status_reason": status_reason,
        "representation": string_repr,
        "arguments": arguments or {},
        "how_generated": how_generated,  # iid, mutation, etc.
        "features": {
            **{
                f"target:{k}".strip(":"): v for k, v in data.target_observations.items()
            },
            **data.events,
        },
        "metadata": {
            **(metadata or {}),
            "traceback": getattr(data.extra_information, "_expected_traceback", None),
        },
        "coverage": coverage,
    }


_WROTE_TO = set()


def _deliver_to_file(value):  # pragma: no cover
    kind = "testcases" if value["type"] == "test_case" else "info"
    fname = storage_directory("observed", f"{date.today().isoformat()}_{kind}.jsonl")
    fname.parent.mkdir(exist_ok=True)
    _WROTE_TO.add(fname)
    with fname.open(mode="a") as f:
        f.write(json.dumps(value) + "\n")


if "HYPOTHESIS_EXPERIMENTAL_OBSERVABILITY" in os.environ:  # pragma: no cover
    TESTCASE_CALLBACKS.append(_deliver_to_file)

    # Remove files more than a week old, to cap the size on disk
    max_age = (date.today() - timedelta(days=8)).isoformat()
    for f in storage_directory("observed").glob("*.jsonl"):
        if f.stem < max_age:  # pragma: no branch
            f.unlink(missing_ok=True)
