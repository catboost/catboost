"""Capture or exactly replay 106 native Metal card 3 snapshot fixtures.

The original 244 compatibility fixtures remain separate. Creation verifies the
published 20260914T000929Z extension, fits a complete model and stops a second fit
after two trees. Replay copies each snapshot into a new directory and compares
all model arrays, evaluation cursors, GPU predictions and metric histories.
Custom objective descriptors are stored as JSON Metal source, never pickle.
"""
import argparse
from collections import Counter
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import time
import traceback


CHECKPOINT = '20260914T000929Z'
EXPECTED_CASES = 106
SCHEMA = 1
RMSE_BODY = '''
const float residual = target - approx;
return float3(-weight * residual * residual, weight * residual, weight);
'''
FILES = ('state.snapshot', 'expected.npz', 'inputs.npz', 'case.json', 'partial.json')


def sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def load_smoke():
    """Load published fixture definitions without importing an installed package."""
    directory = Path(__file__).resolve().parents[1] / 'training_modes_acceptance'
    modules = []
    previous = sys.modules.get('run')
    try:
        for name in ('run', 'smoke'):
            path = directory / (name + '.py')
            spec = importlib.util.spec_from_file_location('card3_baseline_' + name, path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            modules.append(module)
            if name == 'run':
                sys.modules['run'] = module
    finally:
        if previous is None:
            sys.modules.pop('run', None)
        else:
            sys.modules['run'] = previous
    run, smoke = modules
    def load_previous(name):
        # The nested compound helper also imports `run` while being loaded.
        previous_run = sys.modules.get('run')
        try:
            sys.modules['run'] = run
            return run.load_helper(name)
        finally:
            if previous_run is None:
                sys.modules.pop('run', None)
            else:
                sys.modules['run'] = previous_run
    smoke.load_helper = load_previous
    return smoke


def configurations(smoke):
    yield from smoke.configurations()
    for path, category in (
        ('plain-doc', 'numeric'), ('plain-doc', 'ctr'),
        ('plain-feature', 'numeric'), ('plain-feature', 'compound'),
        ('ordered-feature', 'numeric'), ('ordered-feature', 'compound'),
    ):
        yield dict(name=f'custom-{path}-{category}-p4', family='custom',
                   policy='SymmetricTree', boosting='Ordered' if path == 'ordered-feature' else 'Plain',
                   category=category, count=4, loss='RMSE', custom_path=path)


def inventory():
    grid = list(configurations(load_smoke()))
    assert len(grid) == EXPECTED_CASES and len({case['name'] for case in grid}) == len(grid)
    return dict(schema=SCHEMA, baseline_checkpoint=CHECKPOINT, expected_cases=len(grid),
                families=dict(Counter(case['family'] for case in grid)), cases=grid)


class MetalObjective:
    def __init__(self, source):
        self.source = source

    def calc_ders_range_metal(self):
        return self.source

    def calc_ders_range(self, *args):
        raise AssertionError('a baseline fit must not execute a CPU objective')

    def calc_ders_range_gpu(self, *args):
        raise AssertionError('a Metal baseline fit must not execute a CUDA objective')


def fit_options(options):
    result = dict(options)
    descriptor = result.get('loss_function')
    if isinstance(descriptor, dict):
        assert set(descriptor) == {'type', 'source'} and descriptor['type'] == 'metal_scalar_v1'
        assert isinstance(descriptor['source'], str) and descriptor['source'].strip()
        result['loss_function'] = MetalObjective(descriptor['source'])
    assert result['task_type'] == 'GPU'
    return result


def make_case(config, smoke):
    x, y, evaluation, pool_options = smoke.data(config)
    options = smoke.options(config)
    if config['family'] == 'custom':
        options.update(loss_function=dict(type='metal_scalar_v1', source=RMSE_BODY),
                       eval_metric='RMSE', custom_metric=['MAE'], boost_from_average=False,
                       iterations=7, leaf_estimation_iterations=2,
                       leaf_estimation_backtracking='Armijo',
                       data_partition='DocParallel' if config['custom_path'] == 'plain-doc' else 'FeatureParallel')
    values = dict(x=x, evaluation=evaluation, y=y, weights=pool_options['weight'],
                  groups=pool_options['group_id'])
    for key in ('pairs', 'pairs_weight'):
        if key in pool_options:
            values[key] = pool_options[key]
    return dict(config=config, options=options, cats=pool_options['cat_features'], values=values)


def save_inputs(folder, values, cats):
    import numpy as np
    numeric = [i for i in range(values['x'].shape[1]) if i not in cats]
    arrays = {key: values[key] for key in ('y', 'weights', 'groups')}
    arrays.update({key: values[key] for key in ('pairs', 'pairs_weight') if key in values})
    for key in ('x', 'evaluation'):
        arrays[key + '_numeric'] = np.asarray(values[key][:, numeric], np.float32)
        arrays[key + '_categories'] = np.asarray(values[key][:, cats], str)
    np.savez(folder / 'inputs.npz', **arrays)


def load_inputs(folder, cats):
    import numpy as np
    with np.load(folder / 'inputs.npz', allow_pickle=False) as saved:
        values = {key: saved[key] for key in ('y', 'weights', 'groups')}
        for key in ('pairs', 'pairs_weight'):
            if key in saved.files:
                values[key] = saved[key]
        assert ('pairs' in values) == ('pairs_weight' in values)
        for key in ('x', 'evaluation'):
            numeric, categories = saved[key + '_numeric'], saved[key + '_categories']
            columns = numeric.shape[1] + categories.shape[1]
            assert len(cats) == categories.shape[1] and len(set(cats)) == len(cats)
            assert all(0 <= index < columns for index in cats)
            x = np.empty((len(numeric), columns), dtype=object if cats else np.float32)
            numeric_columns = [i for i in range(columns) if i not in cats]
            x[:, numeric_columns], x[:, cats] = numeric, categories
            values[key] = x
    return values


def pools(case):
    from catboost import Pool
    values = case['values']
    options = dict(cat_features=case['cats'], weight=values['weights'], group_id=values['groups'])
    for key in ('pairs', 'pairs_weight'):
        if key in values:
            options[key] = values[key]
    return [Pool(values[key], values['y'], **options) for key in ('x', 'evaluation')]


def arrays(model, case):
    import numpy as np
    result = {key: getattr(model, key)() for key in
              ('get_leaf_values', 'get_leaf_weights', 'get_tree_leaf_counts', 'get_test_eval')}
    result['gpu_predictions'] = model.predict(case['values']['evaluation'], task_type='GPU',
                                               prediction_type='RawFormulaVal')
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


def snapshot_options(case, folder):
    return fit_options(case['options']) | dict(save_snapshot=True, snapshot_interval=0,
        snapshot_file='state.snapshot', allow_writing_files=True, train_dir=str(folder))


def verify_files(baseline, entries):
    names = set()
    for entry in entries:
        name = entry['name']
        assert name not in names and Path(name).name == name and name not in ('.', '..')
        names.add(name)
        assert set(entry['files_sha256']) == set(FILES)
        for file, expected in entry['files_sha256'].items():
            assert sha256(baseline / name / file) == expected, name + ':' + file


def capture(args, report):
    import catboost
    import numpy as np
    from catboost import CatBoost
    manifest_path = args.release / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    assert args.release.name == CHECKPOINT and manifest['checkpoint'] == CHECKPOINT
    assert sha256(Path(catboost.__file__).parent / '_catboost.so') == manifest['standard_extension_sha256']
    report['baseline_manifest_sha256'] = sha256(manifest_path)
    helpers = [Path(__file__).resolve().parents[1] / relative for relative in
               ('training_modes_acceptance/run.py', 'training_modes_acceptance/smoke.py',
                'compound_ctr_acceptance/smoke.py')]
    report['fixture_helpers_sha256'] = {str(path): sha256(path) for path in helpers}
    smoke = load_smoke()
    for config in report['inventory']['cases']:
        case = make_case(config, smoke)
        folder = args.output_dir / config['name']
        folder.mkdir()
        learn, evaluation = pools(case)
        full = CatBoost().set_params(**fit_options(case['options'])).fit(
            learn, eval_set=evaluation, use_best_model=False)
        callback = Iterations(stop=True)
        partial = CatBoost().set_params(**snapshot_options(case, folder)).fit(
            learn, eval_set=evaluation, use_best_model=False, callbacks=[callback])
        assert partial.tree_count_ == 2 and callback.observed == [1, 2]
        partial.save_model(folder / 'partial.json', format='json')
        document = json.loads((folder / 'partial.json').read_text())
        compounds = sum(len(ctr['elements']) > 1 for ctr in document.get('features_info', {}).get('ctrs', []))
        # Some published modes first select a compound after the second tree.
        # Record the partial state instead of imposing a different fixture.
        np.savez(folder / 'expected.npz', **arrays(full, case))
        save_inputs(folder, case['values'], case['cats'])
        metadata = {key: case[key] for key in ('config', 'options', 'cats')}
        metadata['schema'] = SCHEMA
        (folder / 'case.json').write_text(json.dumps(metadata, indent=2) + '\n')
        report['cases'].append(dict(name=config['name'], family=config['family'],
            actual_permutations=full.get_metadata()['metal_permutations'], partial_compound_ctrs=compounds,
            files_sha256={file: sha256(folder / file) for file in FILES}))
        print('PASS card3 baseline capture: ' + config['name'], flush=True)
    for path, expected in report['fixture_helpers_sha256'].items():
        assert sha256(path) == expected, 'fixture helper changed during capture: ' + path
    verify_files(args.output_dir, report['cases'])


def replay(args, report):
    import numpy as np
    from catboost import CatBoost
    source_report_path = args.baseline / 'report.json'
    source_report_hash = sha256(source_report_path)
    source_report = json.loads(source_report_path.read_text())
    assert source_report['status'] == 'passed' and source_report['schema'] == SCHEMA
    assert source_report['baseline_checkpoint'] == CHECKPOINT
    assert source_report['expected_cases'] == len(source_report['cases']) == EXPECTED_CASES
    assert source_report['inventory']['expected_cases'] == EXPECTED_CASES
    assert [case['name'] for case in source_report['inventory']['cases']] == [
        case['name'] for case in source_report['cases']]
    report.update(source_report_sha256=source_report_hash, inventory=source_report['inventory'])
    verify_files(args.baseline, source_report['cases'])
    for entry in source_report['cases']:
        source, folder = args.baseline / entry['name'], args.output_dir / entry['name']
        folder.mkdir()
        case = json.loads((source / 'case.json').read_text())
        assert case['schema'] == SCHEMA and case['config']['name'] == entry['name']
        case['values'] = load_inputs(source, case['cats'])
        learn, evaluation = pools(case)
        shutil.copy2(source / 'state.snapshot', folder / 'state.snapshot')
        callback = Iterations()
        resumed = CatBoost().set_params(**snapshot_options(case, folder)).fit(
            learn, eval_set=evaluation, use_best_model=False, callbacks=[callback])
        assert callback.observed == list(range(3, case['options']['iterations'] + 1)), callback.observed
        actual = arrays(resumed, case)
        with np.load(source / 'expected.npz', allow_pickle=False) as expected:
            assert set(actual) == set(expected.files)
            for key, value in actual.items():
                np.testing.assert_array_equal(value, expected[key], err_msg=entry['name'] + ':' + key)
        report['cases'].append(dict(name=entry['name'], family=entry['family'], exact=True,
                                   source_snapshot_sha256=entry['files_sha256']['state.snapshot']))
        print('PASS exact card3 recovery: ' + entry['name'], flush=True)
    verify_files(args.baseline, source_report['cases'])
    assert sha256(source_report_path) == source_report_hash, 'baseline report changed during replay'


def main():
    metal = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('list', 'capture', 'replay'))
    parser.add_argument('--package', type=Path)
    parser.add_argument('--release', type=Path, default=metal / '.build/releases' / CHECKPOINT)
    parser.add_argument('--baseline', type=Path)
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args()
    if args.command == 'list':
        print(json.dumps(inventory(), indent=2))
        return
    if args.output_dir is None or (args.command == 'replay' and args.baseline is None):
        parser.error('--output-dir is required; replay also requires --baseline')
    for key in ('package', 'release', 'baseline', 'output_dir'):
        value = getattr(args, key)
        if value is not None:
            setattr(args, key, value.resolve())
    if args.command == 'replay':
        assert not args.output_dir.is_relative_to(args.baseline), 'replay output must be outside immutable fixtures'
    if args.package is not None:
        sys.path.insert(0, str(args.package))
    import catboost
    if args.package is not None:
        assert Path(catboost.__file__).resolve().is_relative_to(args.package)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    original = catboost.CatBoost._fit
    def checked(self, *values, **kwargs):
        assert self.get_params().get('task_type') == 'GPU'
        return original(self, *values, **kwargs)
    catboost.CatBoost._fit = checked
    report = dict(command=args.command, status='running', cases=[], schema=SCHEMA,
        baseline_checkpoint=CHECKPOINT, expected_cases=EXPECTED_CASES, python=sys.executable,
        package=str(Path(catboost.__file__).resolve()), source_sha256=sha256(__file__),
        extension_sha256=sha256(Path(catboost.__file__).parent / '_catboost.so'), started_unix=time.time())
    try:
        if args.command == 'capture':
            report['inventory'] = inventory()
        globals()[args.command](args, report)
        assert len(report['cases']) == EXPECTED_CASES
        assert sha256(__file__) == report['source_sha256'], 'runner changed during acceptance'
        report['status'] = 'passed'
    except BaseException:
        report.update(status='failed', error=traceback.format_exc())
        raise
    finally:
        report['elapsed_seconds'] = time.time() - report['started_unix']
        (args.output_dir / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
        print(f"{report['status']}: {len(report['cases'])}/{EXPECTED_CASES} cases; {args.output_dir / 'report.json'}", flush=True)
        catboost.CatBoost._fit = original


if __name__ == '__main__':
    main()
