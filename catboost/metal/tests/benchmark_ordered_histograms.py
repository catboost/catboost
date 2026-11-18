"""Reproducible actual-Metal Ordered before/after workload, with no CPU fit."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from catboost_metal import _ordered, _inference


def workload(rows, features, bins, depth, iterations):
    random = np.random.default_rng(74921)
    data = random.integers(0, bins, (features, rows), dtype=np.uint8)
    x = data.astype(np.float32) / (bins - 1)
    targets = (1.3 * np.sin(4 * x[0]) - 1.8 * x[1] + .6 * (x[2] > .6)
               + .7 * x[3] * x[4] + random.normal(0, .03, rows)).astype(np.float32)
    weights = (2.0 ** random.integers(-2, 3, rows)).astype(np.float32)
    weights[::31] = 0
    cf = np.repeat(np.arange(features, dtype=np.uint32), bins - 1)
    cb = np.tile(np.arange(bins - 1, dtype=np.uint32), features)
    options = dict(iterations=iterations, depth=depth, learning_rate=.12, l2_leaf_reg=3.,
                   permutation_count=4, random_seed=571, sample_weight=weights)
    return data, targets, cf, cb, options


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--library", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    library = args.library or _ordered.build_library()
    _ordered.build_library = lambda: library
    # Warm shader/pipeline initialization outside recorded training timings.
    tiny = workload(64, 5, 8, 2, 1)
    _ordered.train(*tiny[:4], **tiny[4])
    records = []
    for dimensions in ((8192, 8, 16, 4, 20), (32768, 16, 32, 6, 10), (65536, 24, 32, 8, 5)):
        bins, targets, cf, cb, options = workload(*dimensions)
        result, state, times, gpu = None, None, [], []
        for _ in range(args.repeats):
            begin = time.perf_counter()
            with _ordered.Session(bins, targets, cf, cb, **options) as session:
                for _ in range(options["iterations"]):
                    session.step()
                result = session.result()
                state = session.state()
            times.append(time.perf_counter() - begin)
            gpu.append(result.stats["gpu_seconds"])
        prediction = _inference.predict_bins(bins, result.depths, result.split_features,
                                             result.split_bins, result.leaf_values)
        checksum = hashlib.sha256()
        for array in (result.depths, result.split_features, result.split_bins, result.leaf_values):
            checksum.update(array.tobytes())
        prefix = args.output.with_suffix("").name + f"_{dimensions[0]}"
        np.savez(args.output.parent / (prefix + ".npz"), depths=result.depths,
                 split_features=result.split_features, split_bins=result.split_bins,
                 leaf_values=result.leaf_values, leaf_weights=result.leaf_weights,
                 predictions=result.predictions, cursors=state["cursors"])
        records.append(dict(rows=dimensions[0], features=dimensions[1], bins=dimensions[2],
                            depth=dimensions[3], iterations=dimensions[4], candidates=len(cf),
                            wall_seconds=times, gpu_seconds=gpu, model_sha256=checksum.hexdigest(),
                            rmse=[float(result.loss[0]), float(result.loss[-1])],
                            inference_max_error=float(np.max(np.abs(prediction - result.predictions))),
                            depths=result.depths.tolist(), search_permutations=result.stats["search_permutations"],
                            dispatches=result.stats["kernel_dispatches"]))
        print(json.dumps(records[-1]), flush=True)
    args.output.write_text(json.dumps(dict(library=str(library), records=records), indent=2) + "\n")


if __name__ == "__main__":
    main()
