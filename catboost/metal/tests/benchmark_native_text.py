"""Paired native Metal text fits for the PR 3194 slowdown report.

Run the same command in the pre-change and candidate environments. BoW and
numeric controls use the same regression labels and split. NaiveBayes uses
classification as required by CatBoost. Callback timestamps give elapsed
wall time through the first tree and between later tree completions.
"""
import argparse
import hashlib
import json
import platform
import statistics
from pathlib import Path
from time import perf_counter

import numpy as np
from catboost import CatBoostClassifier, CatBoostRegressor, Pool, _catboost


class IterationClock:
    def __init__(self):
        self.times = []

    def after_iteration(self, info):
        self.times.append(perf_counter())
        return True


def run(args):
    rng = np.random.default_rng(413)
    tokens = rng.integers(0, args.vocabulary, (args.rows, 32))
    x = rng.normal(size=(args.rows, 4))
    y = x[:, 0] + 5 * (tokens < args.vocabulary // 2).mean(axis=1) + (tokens == 21).sum(axis=1)
    split = 3 * args.rows // 4
    text_features = None
    classifier = args.case == "naivebayes"
    if classifier:
        y = y > np.median(y[:split])
    input_hashes = {name: hashlib.sha256(array.tobytes()).hexdigest()
                    for name, array in (("token_ids", tokens), ("numeric", x), ("target", y))}
    options = dict(task_type="GPU", depth=args.depth, iterations=args.iterations, border_count=128,
                   learning_rate=.1, random_seed=31, random_strength=0, bootstrap_type="No",
                   thread_count=2, verbose=False, allow_writing_files=False)
    if args.case != "numeric":
        texts = [" ".join(f"tok{v}" for v in doc) for doc in tokens]
        x = np.column_stack([x.astype(object), texts])
        text_features = [4]
        top_tokens = args.vocabulary if args.top_tokens is None else args.top_tokens
        calcer = "NaiveBayes" if classifier else f"BoW:top_tokens_count={top_tokens}"
        options["text_processing"] = dict(
            tokenizers=[dict(tokenizer_id="Space", delimiter=" ")],
            dictionaries=[dict(dictionary_id="Word", occurrence_lower_bound="1")],
            feature_processing={"default": [dict(tokenizers_names=["Space"], dictionaries_names=["Word"],
                                                feature_calcers=[calcer])]},
        )
    learn = Pool(x[:split], y[:split], text_features=text_features)
    valid = Pool(x[split:], y[split:], text_features=text_features)
    observations = []
    for _ in range(args.repeats + 1):
        model = (CatBoostClassifier if classifier else CatBoostRegressor)(**options)
        clock = IterationClock()
        start = perf_counter()
        model.fit(learn, eval_set=valid, use_best_model=False, callbacks=[clock])
        finished = perf_counter()
        assert model.get_metadata()["metal_backend"] == "METAL"
        raw = np.asarray(model.predict(valid, prediction_type="RawFormulaVal"))
        assert np.isfinite(raw).all() and len(clock.times) == args.iterations
        observations.append(dict(fit_seconds=finished-start, through_first_tree=clock.times[0]-start,
                                 later_tree_median=statistics.median(np.diff(clock.times)),
                                 prediction_sha256=hashlib.sha256(raw.tobytes()).hexdigest(),
                                 validation=model.get_evals_result()["validation"]))
    return dict(config=vars(args), input_sha256=input_hashes, learn_rows=split,
                macos=platform.mac_ver()[0], device=model.get_metadata()["metal_device"],
                extension_sha256=hashlib.sha256(Path(_catboost.__file__).read_bytes()).hexdigest(),
                cold=observations[0], warm=observations[1:])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case", choices=("numeric", "bow", "naivebayes"))
    parser.add_argument("--rows", type=int, default=12288)
    parser.add_argument("--vocabulary", type=int, default=384)
    parser.add_argument("--top-tokens", type=int, help="BoW feature limit within the fixed corpus vocabulary")
    parser.add_argument("--iterations", type=int, default=8)
    parser.add_argument("--depth", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=2)
    args = parser.parse_args()
    if args.iterations < 2 or args.repeats < 1 or args.rows < 4 or args.vocabulary < 2:
        parser.error("need at least two iterations, one repetition, four rows and two tokens")
    if args.top_tokens is not None and (args.case != "bow" or not 1 <= args.top_tokens <= args.vocabulary):
        parser.error("--top-tokens requires a bow run and a value between 1 and --vocabulary")
    print(json.dumps(run(args), indent=2))
