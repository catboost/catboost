"""Compare native Metal training against checked-in CUDA canonical histories.

Reproduces test_gpu.test_grow_policies with NO_RANDOM_PARAMS. Canonical outputs
are repository fixtures, not a live NVIDIA run. This diagnostic reports every
difference without silently increasing tolerances or treating coverage as parity.
Requires the rebuilt native CatBoost package in PYTHONPATH.
"""

import argparse
import hashlib
import json
from pathlib import Path
import platform
import subprocess

import numpy as np
from catboost import CatBoostClassifier, CatBoostRegressor, Pool


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--loss", nargs="+", default=["RMSE", "Logloss"])
    parser.add_argument("--scores", nargs="+", default=["L2", "Cosine"])
    parser.add_argument("--output-dir", type=Path,
                        default=Path(__file__).resolve().parents[1] / ".build" / "cuda-fixtures")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    import catboost._catboost as extension
    extension_path = Path(extension.__file__)
    provenance = {
        "extension_path": str(extension_path),
        "extension_sha256": hashlib.sha256(extension_path.read_bytes()).hexdigest(),
        "platform": platform.platform(),
        "upstream_checkout_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
        "local_changes": bool(subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=normal"], cwd=root, text=True).strip()),
        "comparison": "Checked-in CUDA histories; no live NVIDIA execution.",
    }
    (args.output_dir / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    output = []
    for loss in args.loss:
        if loss not in ("RMSE", "Logloss", "MultiClass"):
            parser.error("This reproduction supports RMSE, Logloss, or MultiClass.")
        dataset = "cloudness_small" if loss == "MultiClass" else "adult"
        data = root / "pytest" / "data" / dataset
        for score in args.scores:
            options = dict(
                iterations=20, task_type="GPU", boosting_type="Plain", grow_policy="SymmetricTree",
                loss_function=loss, score_function=score, random_strength=0, bootstrap_type="No",
                has_time=True, use_best_model=False, verbose=False, allow_writing_files=False)
            learn = Pool(str(data / "train_small"), column_description=str(data / "train.cd"))
            test = Pool(str(data / "test_small"), column_description=str(data / "train.cd"))
            cls = CatBoostRegressor if loss == "RMSE" else CatBoostClassifier
            model = cls(**options).fit(learn, eval_set=test)
            if model.get_metadata().get("metal_backend") != "METAL":
                raise AssertionError("The native fit did not identify the Metal backend.")
            canonical = root / "pytest/cuda_tests/canondata" / (
                f"test_gpu.test_grow_policies_{loss}-{score}-SymmetricTree-Plain_")
            results = {}
            for source, key in (("learn_error.tsv", "learn"), ("test_error.tsv", "validation")):
                path = canonical / source
                expected = np.loadtxt(path, skiprows=1)[:, 1]
                actual = np.asarray(model.get_evals_result()[key][loss])
                if expected.shape != actual.shape:
                    raise AssertionError("Metal and CUDA fixture histories have different lengths.")
                difference = np.abs(actual - expected)
                results[key] = {
                    "cuda_fixture": expected.tolist(), "metal": actual.tolist(),
                    "absolute_difference": difference.tolist(), "max_absolute_difference": float(difference.max()),
                    "relative_final_difference": float((actual[-1] - expected[-1]) / expected[-1]),
                    "fixture_path": str(path), "fixture_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                }
            record = {"loss": loss, "score": score, "parameters": model.get_all_params(), "histories": results,
                      "data_sha256": {name: hashlib.sha256((data / name).read_bytes()).hexdigest()
                                      for name in ("train_small", "test_small", "train.cd")},
                      "extension_sha256": provenance["extension_sha256"]}
            output.append(record)
            (args.output_dir / "results.json").write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")
            print(json.dumps({"loss": loss, "score": score,
                "cuda_final": results["validation"]["cuda_fixture"][-1],
                "metal_final": results["validation"]["metal"][-1],
                "max_difference": results["validation"]["max_absolute_difference"]}), flush=True)


if __name__ == "__main__":
    main()
