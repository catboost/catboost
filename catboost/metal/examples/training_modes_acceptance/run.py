"""Capture and replay old installed native Metal training-mode snapshots."""
import argparse
import hashlib
import importlib.util
from itertools import product
import json
from pathlib import Path
import shutil
import sys
import time
import traceback


def sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def load_helper(name):
    path = Path(__file__).resolve().parents[1] / 'compound_ctr_acceptance' / name
    spec = importlib.util.spec_from_file_location('previous_' + path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, path


def cases():
    import numpy as np
    baseline, _ = load_helper('run.py')
    compound, _ = load_helper('smoke.py')
    for boosting, category, count in product(('Plain', 'Ordered'), ('numeric', 'onehot', 'ctr', 'compound'), (1, 4)):
        if category == 'compound':
            x, y, _, weights = compound.data(2)
            evaluation = x.copy()
            evaluation[::17, 0] = 'unseen'
            values = dict(x=x, evaluation=evaluation, y=y, weights=weights)
            options = compound.config(boosting, count, 2, 'RMSE')
            cats = [0, 1]
        else:
            values = baseline.inputs(category)
            options = baseline.config(boosting, category, count)
            cats = [] if category == 'numeric' else [2, 3]
        yield dict(name=f'{boosting}-{category}-p{count}', category=category, cats=cats,
                   options=options, values=values, ranker=False)
    for policy, count in product(('Depthwise', 'Lossguide', 'Region'), (1, 4)):
        options = baseline.config('Plain', 'ctr', count)
        options['grow_policy'] = policy
        if policy == 'Lossguide':
            options['max_leaves'] = 6
        yield dict(name=f'{policy}-RMSE-ctr-p{count}', category='ctr', cats=[2, 3],
                   options=options, values=baseline.inputs('ctr'), ranker=False)
    for loss, count in product(('QueryRMSE', 'QuerySoftMax', 'YetiRank', 'YetiRankPairwise'), (1, 4)):
        rng = np.random.default_rng(34721)
        sizes = np.tile([3, 5, 8, 7, 12, 4, 9, 16], 3)
        rows = sizes.sum()
        groups = np.repeat(np.arange(len(sizes)), sizes)
        category = rng.choice(12, rows, p=np.arange(1, 13) / 78)
        x = np.array([[f'category-{c}'] for c in category], object)
        y = (category % 2).astype(np.float32)
        weights = rng.uniform(.2, 2, rows).astype(np.float32)
        weights[::17] = 0
        options = dict(task_type='GPU', loss_function=loss, iterations=5, depth=2, learning_rate=.17,
                       l2_leaf_reg=2, bootstrap_type='No', random_strength=.2, score_function='L2',
                       model_size_reg=0, boost_from_average=False, leaf_estimation_method='Gradient',
                       leaf_estimation_iterations=2, leaf_estimation_backtracking='No',
                       one_hot_max_size=1, max_ctr_complexity=1,
                       simple_ctr=['Borders:CtrBorderType=Uniform:CtrBorderCount=1:Prior=0.5'],
                       ctr_target_border_count=1, ctr_history_unit='Group', permutation_count=count,
                       has_time=count == 1, random_seed=735, verbose=False, allow_writing_files=False)
        if loss.startswith('Yeti'):
            options.update(loss_function=loss + ':permutations=7;decay=.85', score_function='NewtonL2',
                           leaf_estimation_method='Newton', leaf_estimation_iterations=3)
            if loss == 'YetiRankPairwise':
                options['bayesian_matrix_reg'] = .2
        evaluation = x.copy()
        evaluation[::17, 0] = 'unseen'
        yield dict(name=f'SymmetricTree-{loss}-ctr-p{count}', category='ctr', cats=[0],
                   options=options, ranker=True,
                   values=dict(x=x, evaluation=evaluation, y=y, weights=weights, groups=groups))


def save_inputs(folder, values, cats):
    import numpy as np
    numeric = [i for i in range(values['x'].shape[1]) if i not in cats]
    arrays = {key: values[key] for key in ('y', 'weights')}
    arrays['groups'] = values.get('groups', np.array([], np.uint64))
    for key in ('x', 'evaluation'):
        arrays[key + '_numeric'] = np.asarray(values[key][:, numeric], np.float32)
        arrays[key + '_categories'] = np.asarray(values[key][:, cats], str)
    np.savez(folder / 'inputs.npz', **arrays)


def load_inputs(folder, cats):
    import numpy as np
    with np.load(folder / 'inputs.npz', allow_pickle=False) as saved:
        values = {key: saved[key] for key in ('y', 'weights', 'groups')}
        for key in ('x', 'evaluation'):
            numeric, categories = saved[key + '_numeric'], saved[key + '_categories']
            columns = numeric.shape[1] + categories.shape[1]
            x = np.empty((len(numeric), columns), dtype=object if cats else np.float32)
            numeric_columns = [i for i in range(columns) if i not in cats]
            x[:, numeric_columns], x[:, cats] = numeric, categories
            values[key] = x
    return values


def pools(case):
    from catboost import Pool
    values = case['values']
    options = dict(cat_features=case['cats'], weight=values['weights'])
    if len(values.get('groups', [])):
        options['group_id'] = values['groups']
    return [Pool(values[key], values['y'], **options) for key in ('x', 'evaluation')]


def arrays(model, case):
    import numpy as np
    result = {key: getattr(model, key)() for key in
              ('get_leaf_values', 'get_leaf_weights', 'get_tree_leaf_counts', 'get_test_eval')}
    prediction_options = {} if case['ranker'] else dict(prediction_type='RawFormulaVal')
    result['gpu_predictions'] = model.predict(case['values']['evaluation'], task_type='GPU', **prediction_options)
    result['history'] = np.array(json.dumps(model.get_evals_result(), sort_keys=True))
    assert model.get_metadata()['metal_backend'] == 'METAL'
    assert model.get_all_params()['task_type'] == 'GPU'
    assert model.tree_count_ == case['options']['iterations']
    assert np.isfinite(result['gpu_predictions']).all()
    return result


class Iterations:
    def __init__(self, stop=False):
        self.stop, self.observed = stop, []

    def after_iteration(self, info):
        self.observed.append(info.iteration)
        return not self.stop or info.iteration < 2


def baseline_create(args, report):
    import catboost
    import numpy as np
    from catboost import CatBoostRanker, CatBoostRegressor
    manifest = json.loads((args.release / 'manifest.json').read_text())
    assert sha256(Path(catboost.__file__).parent / '_catboost.so') == manifest['standard_extension_sha256']
    report['baseline_checkpoint'] = args.release.name
    report['fixture_helpers_sha256'] = {str(path): sha256(path) for _, path in
                                      (load_helper('run.py'), load_helper('smoke.py'))}
    for case in cases():
        folder = args.output_dir / case['name']
        folder.mkdir()
        cls = CatBoostRanker if case['ranker'] else CatBoostRegressor
        learn, evaluation = pools(case)
        full = cls().set_params(**case['options']).fit(learn, eval_set=evaluation, use_best_model=False)
        options = case['options'] | dict(save_snapshot=True, snapshot_interval=0, snapshot_file='state.snapshot',
                                        allow_writing_files=True, train_dir=str(folder))
        partial = cls().set_params(**options).fit(learn, eval_set=evaluation, use_best_model=False,
                                                callbacks=[Iterations(stop=True)])
        assert partial.tree_count_ == 2
        partial.save_model(folder / 'partial.json', format='json')
        document = json.loads((folder / 'partial.json').read_text())
        compound_count = sum(len(c['elements']) > 1 for c in document.get('features_info', {}).get('ctrs', []))
        if case['category'] == 'compound':
            assert compound_count > 0, case['name'] + ': expected compound state before snapshot'
        np.savez(folder / 'expected.npz', **arrays(full, case))
        save_inputs(folder, case['values'], case['cats'])
        metadata = {key: case[key] for key in ('options', 'cats', 'ranker', 'category')}
        (folder / 'case.json').write_text(json.dumps(metadata, indent=2) + '\n')
        files = ('state.snapshot', 'expected.npz', 'inputs.npz', 'case.json', 'partial.json')
        report['cases'].append(dict(name=case['name'], actual_permutations=full.get_metadata()['metal_permutations'],
                                   partial_compound_ctrs=compound_count,
                                   files_sha256={file: sha256(folder / file) for file in files}))
        print('PASS old220233Z baseline:', case['name'], flush=True)
    assert len(report['cases']) == 30


def baseline_replay(args, report):
    import numpy as np
    from catboost import CatBoostRanker, CatBoostRegressor
    if args.baseline is None:
        raise ValueError('--baseline is required')
    source_report = json.loads((args.baseline / 'report.json').read_text())
    assert source_report['status'] == 'passed' and len(source_report['cases']) == 30
    report.update(baseline_checkpoint=source_report['baseline_checkpoint'],
                  source_report_sha256=sha256(args.baseline / 'report.json'))
    for entry in source_report['cases']:
        for name, expected in entry['files_sha256'].items():
            assert sha256(args.baseline / entry['name'] / name) == expected
    for entry in source_report['cases']:
        source, folder = args.baseline / entry['name'], args.output_dir / entry['name']
        folder.mkdir()
        case = json.loads((source / 'case.json').read_text())
        case['values'] = load_inputs(source, case['cats'])
        learn, evaluation = pools(case)
        cls = CatBoostRanker if case['ranker'] else CatBoostRegressor
        shutil.copy2(source / 'state.snapshot', folder / 'state.snapshot')
        options = case['options'] | dict(save_snapshot=True, snapshot_interval=0, snapshot_file='state.snapshot',
                                        allow_writing_files=True, train_dir=str(folder))
        callback = Iterations()
        resumed = cls().set_params(**options).fit(learn, eval_set=evaluation, use_best_model=False, callbacks=[callback])
        assert callback.observed == list(range(3, case['options']['iterations'] + 1)), callback.observed
        actual = arrays(resumed, case)
        with np.load(source / 'expected.npz', allow_pickle=False) as expected:
            assert set(actual) == set(expected.files)
            for key, value in actual.items():
                np.testing.assert_array_equal(value, expected[key], err_msg=entry['name'] + ':' + key)
        report['cases'].append(dict(name=entry['name'], exact=True,
                                   source_snapshot_sha256=entry['files_sha256']['state.snapshot']))
        print('PASS exact220233Z recovery:', entry['name'], flush=True)
    for entry in source_report['cases']:
        for name, expected in entry['files_sha256'].items():
            assert sha256(args.baseline / entry['name'] / name) == expected


def main():
    metal = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('baseline-create', 'baseline-replay'))
    parser.add_argument('--package', type=Path)
    parser.add_argument('--release', type=Path, default=metal / '.build/releases/20260913T220233Z')
    parser.add_argument('--baseline', type=Path)
    parser.add_argument('--output-dir', type=Path, required=True)
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
    original = catboost.CatBoost._fit
    def checked(self, *values, **kwargs):
        assert self.get_params().get('task_type') == 'GPU'
        return original(self, *values, **kwargs)
    catboost.CatBoost._fit = checked
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
        catboost.CatBoost._fit = original


if __name__ == '__main__':
    main()
