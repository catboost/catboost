"""Capture and replay exact pre-compound native Metal snapshots.

All fits use GPU. Baseline capture requires the extension hash recorded by
the preserved release. Each command requires a fresh output directory.
"""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time
import traceback


def sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def inputs(category):
    import numpy as np
    rng = np.random.default_rng(739031)
    rows = 192
    numeric = rng.normal(size=(rows, 2)).astype(np.float32)
    codes = rng.integers(0, 5, size=(rows, 2))
    labels = (numeric[:, 0] + .4 * numeric[:, 1] + 1.7 * (codes[:, 0] == 2)
              - .8 * (codes[:, 1] == 1)).astype(np.float32)
    weights = rng.uniform(.3, 1.8, rows).astype(np.float32)
    weights[::19] = 0
    categories = np.array([['c' + str(a), 'd' + str(b)] for a, b in codes])
    x = numeric if category == 'numeric' else np.column_stack((numeric.astype(object), categories))
    evaluation = x.copy()
    evaluation[:, 0] = np.asarray(evaluation[:, 0], float) + .1
    if category != 'numeric':
        evaluation[::17, 2] = 'unseen'
    return dict(x=x, evaluation=evaluation, numeric=numeric, categories=categories,
                y=labels, weights=weights)


def config(boosting, category, count):
    result = dict(task_type='GPU', loss_function='RMSE', iterations=5, depth=3,
                  boosting_type=boosting, grow_policy='SymmetricTree', learning_rate=.17,
                  leaf_estimation_method='Newton', leaf_estimation_iterations=3,
                  leaf_estimation_backtracking='No', bootstrap_type='Bernoulli', subsample=.8,
                  l2_leaf_reg=2, random_seed=43, random_strength=.3, border_count=8,
                  score_function='Cosine', permutation_count=count, has_time=False,
                  one_hot_max_size=6 if category == 'onehot' else 1,
                  max_ctr_complexity=1, verbose=False, allow_writing_files=False)
    if category == 'ctr':
        result.update(simple_ctr=['Borders:CtrBorderCount=7:TargetBorderCount=1:Prior=0.5'],
                      ctr_target_border_count=1, counter_calc_method='SkipTest')
    if boosting == 'Ordered':
        result.update(min_fold_size=16, fold_len_multiplier=1.7, fold_permutation_block=3)
    return result


def pools(values, category):
    from catboost import Pool
    cats = [] if category == 'numeric' else [2, 3]
    return [Pool(values[key], values['y'], weight=values['weights'], cat_features=cats)
            for key in ('x', 'evaluation')]


def arrays(model, evaluation):
    import numpy as np
    result = {key: getattr(model, key)() for key in
              ('get_leaf_values', 'get_leaf_weights', 'get_tree_leaf_counts', 'get_test_eval')}
    result['gpu_predictions'] = model.predict(evaluation, prediction_type='RawFormulaVal', task_type='GPU')
    result['history'] = np.array(json.dumps(model.get_evals_result(), sort_keys=True))
    assert model.get_metadata()['metal_backend'] == 'METAL'
    assert model.get_all_params()['task_type'] == 'GPU'
    assert model.tree_count_ == 5
    assert np.isfinite(result['gpu_predictions']).all()
    return result


class StopAfterTwo:
    def after_iteration(self, info):
        return info.iteration < 2


class ResumedIterations:
    def __init__(self):
        self.iterations = []

    def after_iteration(self, info):
        self.iterations.append(info.iteration)
        return True

    def verify(self):
        assert self.iterations == [3, 4, 5], self.iterations


def baseline_create(args, report):
    import catboost
    import numpy as np
    from catboost import CatBoostRegressor
    manifest = json.loads((args.release / 'manifest.json').read_text())
    assert sha256(Path(catboost.__file__).parent / '_catboost.so') == manifest['standard_extension_sha256']
    report['baseline_checkpoint'] = args.release.name
    for boosting in ('Plain', 'Ordered'):
        for category in ('numeric', 'onehot', 'ctr'):
            for count in (1, 4):
                name = f'{boosting}-{category}-p{count}'
                folder = args.output_dir / name
                folder.mkdir()
                values = inputs(category)
                learn, evaluation = pools(values, category)
                options = config(boosting, category, count)
                full = CatBoostRegressor().set_params(**options).fit(learn, eval_set=evaluation, use_best_model=False)
                saved = options | dict(save_snapshot=True, snapshot_interval=0, snapshot_file='state.snapshot',
                                       allow_writing_files=True, train_dir=str(folder))
                partial = CatBoostRegressor().set_params(**saved).fit(
                    learn, eval_set=evaluation, use_best_model=False, callbacks=[StopAfterTwo()])
                assert partial.tree_count_ == 2
                np.savez(folder / 'expected.npz', **arrays(full, values['evaluation']))
                np.savez(folder / 'inputs.npz', **{key: values[key] for key in ('numeric', 'categories', 'y', 'weights')})
                (folder / 'config.json').write_text(json.dumps(options, indent=2) + '\n')
                files = ('state.snapshot', 'expected.npz', 'inputs.npz', 'config.json')
                report['cases'].append(dict(name=name, category=category, boosting_type=boosting,
                    permutation_count=count, actual_permutations=full.get_metadata()['metal_permutations'],
                    files_sha256={file: sha256(folder / file) for file in files}))
                print('PASS old installed snapshot:', name, flush=True)
    assert len(report['cases']) == 12


def baseline_replay(args, report):
    import numpy as np
    from catboost import CatBoostRegressor
    if args.baseline is None:
        raise ValueError('--baseline is required')
    source_report = json.loads((args.baseline / 'report.json').read_text())
    assert source_report['status'] == 'passed' and len(source_report['cases']) == 12
    report.update(baseline_checkpoint=source_report['baseline_checkpoint'],
                  source_report_sha256=sha256(args.baseline / 'report.json'))
    for case in source_report['cases']:
        source = args.baseline / case['name']
        for file, expected in case['files_sha256'].items():
            assert sha256(source / file) == expected, (case['name'], file)
    for case in source_report['cases']:
        source = args.baseline / case['name']
        folder = args.output_dir / case['name']
        folder.mkdir()
        with np.load(source / 'inputs.npz', allow_pickle=False) as saved:
            values = {key: saved[key] for key in saved.files}
        values['x'] = values['numeric'] if case['category'] == 'numeric' else np.column_stack(
            (values['numeric'].astype(object), values['categories']))
        values['evaluation'] = values['x'].copy()
        values['evaluation'][:, 0] = np.asarray(values['evaluation'][:, 0], float) + .1
        if case['category'] != 'numeric':
            values['evaluation'][::17, 2] = 'unseen'
        learn, evaluation = pools(values, case['category'])
        shutil.copy2(source / 'state.snapshot', folder / 'state.snapshot')
        options = json.loads((source / 'config.json').read_text())
        options.update(save_snapshot=True, snapshot_interval=0, snapshot_file='state.snapshot',
                       allow_writing_files=True, train_dir=str(folder))
        resumed = ResumedIterations()
        model = CatBoostRegressor().set_params(**options).fit(
            learn, eval_set=evaluation, use_best_model=False, callbacks=[resumed])
        resumed.verify()
        actual = arrays(model, values['evaluation'])
        with np.load(source / 'expected.npz', allow_pickle=False) as expected:
            assert set(actual) == set(expected.files)
            for key, value in actual.items():
                np.testing.assert_array_equal(value, expected[key], err_msg=case['name'] + ':' + key)
        report['cases'].append(dict(name=case['name'], exact=True,
                                   source_snapshot_sha256=case['files_sha256']['state.snapshot']))
        print('PASS exact old installed recovery:', case['name'], flush=True)
    for case in source_report['cases']:
        for file, expected in case['files_sha256'].items():
            assert sha256(args.baseline / case['name'] / file) == expected


def main():
    metal = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('baseline-create', 'baseline-replay'))
    parser.add_argument('--package', type=Path)
    parser.add_argument('--release', type=Path, default=metal / '.build/releases/20260913T195124Z')
    parser.add_argument('--baseline', type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    args = parser.parse_args()
    for key in ('package', 'release', 'baseline', 'output_dir'):
        value = getattr(args, key)
        if value is not None:
            setattr(args, key, value.absolute())
    if args.package is not None:
        sys.path.insert(0, str(args.package))
    import catboost
    if args.package is not None:
        assert Path(catboost.__file__).resolve().is_relative_to(args.package.resolve())
    args.output_dir.mkdir(parents=True, exist_ok=False)
    original_fit = catboost.CatBoost._fit
    def checked_fit(self, *values, **kwargs):
        assert self.get_params().get('task_type') == 'GPU'
        return original_fit(self, *values, **kwargs)
    catboost.CatBoost._fit = checked_fit
    report = dict(command=args.command, status='running', cases=[], python=sys.executable,
                  package=str(Path(catboost.__file__).resolve()), source_sha256=sha256(__file__),
                  extension_sha256=sha256(Path(catboost.__file__).parent / '_catboost.so'), started_unix=time.time())
    try:
        globals()[args.command.replace('-', '_')](args, report)
        report['status'] = 'passed'
    except BaseException:
        report.update(status='failed', error=traceback.format_exc())
        raise
    finally:
        report['elapsed_seconds'] = time.time() - report['started_unix']
        (args.output_dir / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
        print(f"{report['status']}: {len(report['cases'])} cases; {args.output_dir / 'report.json'}", flush=True)
        catboost.CatBoost._fit = original_fit


if __name__ == '__main__':
    main()
