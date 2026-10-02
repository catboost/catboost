"""Isolated CLI, installed-package and old-snapshot Metal release checks.

Every native fit requires task_type='GPU'. No packages are installed and no
existing output directory is reused. Run --help for the individual commands.
"""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback


POLICIES = ('Depthwise', 'Lossguide', 'Region')
CATEGORIES = ('numeric', 'onehot', 'ctr')


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def data(category='numeric'):
    import numpy as np
    rng = np.random.default_rng(71349)
    numeric = rng.normal(size=(120, 3)).astype(np.float32)
    code = np.arange(120) % 6
    signal = numeric[:, 0] + .7 * numeric[:, 1] + .8 * (code == 2)
    groups = np.repeat(np.arange(20), 6)
    edges = []
    for start in range(0, 120, 6):
        order = np.argsort(signal[start:start + 6]) + start
        edges.extend((int(order[i + 1]), int(order[i])) for i in range(5))
    x = numeric if category == 'numeric' else np.column_stack((numeric.astype(object), ['cat' + str(i) for i in code]))
    return x, signal.astype(np.float32), groups, np.array(edges, np.uint32), np.linspace(.4, 2, len(edges), dtype=np.float32)


def config(policy, category, method='Newton', loss='PairLogit'):
    result = dict(task_type='GPU', loss_function=loss, iterations=5, depth=3,
                  grow_policy=policy, learning_rate=.2, l2_leaf_reg=2,
                  score_function='Cosine', leaf_estimation_method=method,
                  leaf_estimation_iterations=4, leaf_estimation_backtracking='Armijo',
                  random_seed=45, random_strength=.2, bootstrap_type='Bernoulli',
                  subsample=.8, border_count=8, one_hot_max_size=8 if category == 'onehot' else 1,
                  verbose=False, allow_writing_files=False)
    if policy == 'Lossguide':
        result['max_leaves'] = 6
    if category == 'ctr':
        result.update(max_ctr_complexity=1, permutation_count=4, ctr_target_border_count=1,
                      simple_ctr=['Borders:CtrBorderCount=7:TargetBorderCount=1:Prior=0.5'],
                      counter_calc_method='SkipTest')
    return result


def save_inputs(path, values):
    import numpy as np
    x, y, groups, edges, weights = values
    np.savez(path, numeric=np.asarray(x[:, :3], np.float32),
             categories=np.asarray(x[:, 3], str) if x.shape[1] == 4 else np.array([], dtype=str),
             y=y, groups=groups, edges=edges, weights=weights)


def load_inputs(path):
    import numpy as np
    with np.load(path, allow_pickle=False) as saved:
        x = saved['numeric']
        if saved['categories'].size:
            x = np.column_stack((x.astype(object), saved['categories']))
        return x, saved['y'], saved['groups'], saved['edges'], saved['weights']


def native_arrays(model, x):
    import numpy as np
    from catboost import CatBoostRanker
    arrays = {key: getattr(model, key)() for key in ('get_leaf_values', 'get_leaf_weights', 'get_tree_leaf_counts', 'get_test_eval')}
    prediction_options = {} if isinstance(model, CatBoostRanker) else dict(prediction_type='RawFormulaVal')
    arrays['gpu_predictions'] = model.predict(x, task_type='GPU', **prediction_options)
    arrays['history'] = np.array(json.dumps(model.get_evals_result(), sort_keys=True))
    assert model.get_metadata()['metal_backend'] == 'METAL'
    assert model.get_all_params()['task_type'] == 'GPU'
    assert np.isfinite(arrays['gpu_predictions']).all()
    return arrays


def assert_exact(arrays, expected_path, name):
    import numpy as np
    with np.load(expected_path, allow_pickle=False) as expected:
        assert set(arrays) == set(expected.files), name + ': array keys changed'
        for key, value in arrays.items():
            np.testing.assert_array_equal(value, expected[key], err_msg=name + ':' + key)


def readers(model, x, folder):
    import numpy as np
    from catboost import CatBoostRanker
    expected = model.predict(x, task_type='GPU')
    assert np.isfinite(expected).all()
    np.testing.assert_allclose(model.predict(x), expected, rtol=4e-6, atol=1e-6)
    for fmt in ('cbm', 'json'):
        path = folder / ('export.' + fmt)
        model.save_model(path, format=fmt)
        restored = CatBoostRanker().load_model(path, format=fmt)
        assert restored.get_metadata()['metal_backend'] == 'METAL'
        np.testing.assert_allclose(restored.predict(x, task_type='GPU'), expected, rtol=4e-6, atol=1e-6)
        np.testing.assert_allclose(restored.predict(x), expected, rtol=4e-6, atol=1e-6)
    return expected


def smoke(args, report):
    import numpy as np
    from catboost import CatBoostRanker, Pool
    for policy in POLICIES:
        for category in CATEGORIES:
            for method in ('Newton', 'Gradient'):
                name = f'{policy}-{category}-{method}'
                folder = args.output_dir / name
                folder.mkdir()
                x, y, groups, edges, weights = data(category)
                pool = Pool(x, y, cat_features=[] if category == 'numeric' else [3],
                            group_id=groups, pairs=edges, pairs_weight=weights)
                model = CatBoostRanker().set_params(**config(policy, category, method)).fit(pool, eval_set=pool, use_best_model=False)
                arrays = native_arrays(model, x)
                prediction = readers(model, x, folder)
                expected_loss = np.average(np.logaddexp(0, prediction[edges[:, 1]] - prediction[edges[:, 0]]), weights=weights)
                np.testing.assert_allclose(model.get_evals_result()['validation']['PairLogit'][-1], expected_loss, rtol=4e-6, atol=1e-7)
                assert model.tree_count_ == 5
                assert model.get_metadata()['metal_permutations'] == ('4' if category == 'ctr' else '1')
                np.savez(folder / 'predictions.npz', **arrays)
                report['cases'].append(dict(name=name, trees=model.tree_count_, validation_pairlogit=float(expected_loss)))
                print('PASS installed GPU:', name, flush=True)


def cli(args, report):
    import numpy as np
    from catboost import CatBoostRanker, Pool
    if args.cli is None or not args.cli.is_file():
        raise ValueError('--cli must name the rebuilt CatBoost executable')
    report['cli'] = str(args.cli)
    report['cli_sha256'] = sha256(args.cli)
    for policy in POLICIES:
        for category in CATEGORIES:
            for method in ('Newton', 'Gradient'):
                name = f'{policy}-{category}-{method}'
                folder = args.output_dir / name
                folder.mkdir()
                x, y, groups, edges, weights = data(category)
                learn, cd, pairs, cbm = (folder / path for path in ('learn.tsv', 'columns.cd', 'pairs.tsv', 'model.cbm'))
                learn.write_text(''.join('\t'.join(map(str, [y[i], groups[i], *x[i]])) + '\n' for i in range(len(y))))
                cd.write_text('0\tLabel\n1\tGroupId\n' + ''.join(f'{f + 2}\t' + ('Categ' if category != 'numeric' and f == 3 else 'Num') + f'\tfeature{f}\n' for f in range(x.shape[1])))
                pairs.write_text(''.join(f'{w}\t{l}\t{weight}\n' for (w, l), weight in zip(edges, weights)))
                command = [str(args.cli), 'fit', '--task-type', 'GPU', '--grow-policy', policy,
                           '--loss-function', 'PairLogit', '--leaf-estimation-method', method,
                           '--leaf-estimation-iterations', '4', '--leaf-estimation-backtracking', 'Armijo',
                           '--learn-set', str(learn), '--test-set', str(learn), '--learn-pairs', str(pairs),
                           '--test-pairs', str(pairs), '--column-description', str(cd),
                           '--one-hot-max-size', '8' if category == 'onehot' else '1',
                           '--max-ctr-complexity', '1', '--simple-ctr', 'Borders:CtrBorderCount=7:TargetBorderCount=1:Prior=0.5',
                           '--counter-calc-method', 'SkipTest', '--permutations', '4', '--iterations', '5',
                           '--depth', '3', '--bootstrap-type', 'Bernoulli', '--subsample', '.8',
                           '--random-strength', '.2', '--use-best-model', 'false', '--allow-writing-files', 'false',
                           '--model-file', str(cbm)]
                if policy == 'Lossguide':
                    command += ['--max-leaves', '6']
                (folder / 'command.json').write_text(json.dumps(command, indent=2) + '\n')
                with (folder / 'fit.log').open('w') as log:
                    subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
                model = CatBoostRanker().load_model(cbm)
                pool = Pool(str(learn), column_description=str(cd), pairs=str(pairs))
                prediction = readers(model, pool, folder)
                assert model.tree_count_ == 5
                assert model.get_metadata()['metal_backend'] == 'METAL'
                assert model.get_metadata()['metal_permutations'] == ('4' if category == 'ctr' else '1')
                np.savez(folder / 'predictions.npz', predictions=prediction)
                report['cases'].append(dict(name=name, trees=model.tree_count_))
                print('PASS CLI GPU:', name, flush=True)


class StopAfter:
    def after_iteration(self, info):
        return info.iteration < 2


class ResumedIterations:
    """Native callbacks run for newly trained trees, not snapshot replay."""
    def __init__(self):
        self.iterations = []

    def after_iteration(self, info):
        self.iterations.append(info.iteration)
        return True

    def verify(self):
        assert self.iterations == [3, 4, 5], f'expected two restored trees; new training callbacks were {self.iterations}'


def baseline_create(args, report):
    """Save old native symmetric PairLogit and scalar greedy reference fits."""
    import numpy as np
    import catboost
    from catboost import CatBoostRanker, CatBoostRegressor, Pool
    manifest = json.loads((args.release / 'manifest.json').read_text())
    extension = Path(catboost.__file__).parent / '_catboost.so'
    assert sha256(extension) == manifest['standard_extension_sha256'], 'baseline must use the preserved checkpoint wheel'
    report['baseline_checkpoint'] = args.release.name
    for loss, policy in [('PairLogit', 'SymmetricTree'), *(('RMSE', p) for p in POLICIES)]:
        for category in ('numeric', 'ctr'):
            name = f'{loss}-{policy}-{category}'
            folder = args.output_dir / name
            folder.mkdir()
            x, y, groups, edges, weights = data(category)
            config_ = config(policy, category, loss=loss)
            pool = Pool(x, y, cat_features=[] if category == 'numeric' else [3], group_id=groups,
                        **(dict(pairs=edges, pairs_weight=weights) if loss == 'PairLogit' else {}))
            cls = CatBoostRanker if loss == 'PairLogit' else CatBoostRegressor
            full = cls().set_params(**config_).fit(pool, eval_set=pool, use_best_model=False)
            saved = config_ | dict(save_snapshot=True, snapshot_interval=0, snapshot_file='state.snapshot',
                                   allow_writing_files=True, train_dir=str(folder))
            partial = cls().set_params(**saved).fit(pool, eval_set=pool, use_best_model=False, callbacks=[StopAfter()])
            assert partial.tree_count_ == 2
            np.savez(folder / 'expected.npz', **native_arrays(full, x))
            (folder / 'config.json').write_text(json.dumps(config_, indent=2) + '\n')
            save_inputs(folder / 'inputs.npz', (x, y, groups, edges, weights))
            report['cases'].append(dict(name=name, snapshot_sha256=sha256(folder / 'state.snapshot'),
                                        expected_sha256=sha256(folder / 'expected.npz'),
                                        config_sha256=sha256(folder / 'config.json'),
                                        inputs_sha256=sha256(folder / 'inputs.npz')))
            print('PASS old checkpoint baseline:', name, flush=True)


def baseline_replay(args, report):
    from catboost import CatBoostRanker, CatBoostRegressor, Pool
    if args.baseline is None:
        raise ValueError('--baseline must name baseline-create output')
    source_report = json.loads((args.baseline / 'report.json').read_text())
    assert source_report['status'] == 'passed' and len(source_report['cases']) == 8
    report['baseline_checkpoint'] = source_report['baseline_checkpoint']
    report['source_report_sha256'] = sha256(args.baseline / 'report.json')
    for case in source_report['cases']:
        name = case['name']
        loss, policy, category = name.split('-')
        source = args.baseline / name
        folder = args.output_dir / name
        folder.mkdir()
        assert sha256(source / 'state.snapshot') == case['snapshot_sha256']
        assert sha256(source / 'expected.npz') == case['expected_sha256']
        assert sha256(source / 'config.json') == case['config_sha256']
        assert sha256(source / 'inputs.npz') == case['inputs_sha256']
        shutil.copy2(source / 'state.snapshot', folder / 'state.snapshot')
        config_ = json.loads((source / 'config.json').read_text())
        x, y, groups, edges, weights = load_inputs(source / 'inputs.npz')
        pool = Pool(x, y, cat_features=[] if category == 'numeric' else [3], group_id=groups,
                    **(dict(pairs=edges, pairs_weight=weights) if loss == 'PairLogit' else {}))
        cls = CatBoostRanker if loss == 'PairLogit' else CatBoostRegressor
        saved = config_ | dict(save_snapshot=True, snapshot_interval=0, snapshot_file='state.snapshot',
                               allow_writing_files=True, train_dir=str(folder))
        resumed = ResumedIterations()
        model = cls().set_params(**saved).fit(pool, eval_set=pool, use_best_model=False, callbacks=[resumed])
        resumed.verify()
        assert_exact(native_arrays(model, x), source / 'expected.npz', name)
        report['cases'].append(dict(name=name, exact=True, source_snapshot_sha256=case['snapshot_sha256']))
        print('PASS checkpoint snapshot exact:', name, flush=True)


def main():
    metal = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('cli', 'smoke', 'legacy-replay', 'baseline-create', 'baseline-replay'))
    parser.add_argument('--package', type=Path, help='Directory containing the selected native catboost package; otherwise use PYTHONPATH/current environment')
    parser.add_argument('--standalone', type=Path, help='Directory containing catboost_metal; otherwise use PYTHONPATH/current environment')
    parser.add_argument('--cli', type=Path)
    parser.add_argument('--release', type=Path, default=metal / '.build/releases/20260913T155047Z')
    parser.add_argument('--baseline', type=Path, help='Preserved baseline-create output to replay')
    parser.add_argument('--output-dir', type=Path, required=True, help='New directory; existing paths are rejected')
    args = parser.parse_args()
    for name in ('package', 'standalone', 'cli', 'release', 'baseline', 'output_dir'):
        value = getattr(args, name)
        if value is not None:
            setattr(args, name, value.resolve())
    for package in (args.standalone, args.package):
        if package is not None:
            if not package.is_dir():
                parser.error(f'package directory does not exist: {package}')
            sys.path.insert(0, str(package))
    args.output_dir.mkdir(parents=True, exist_ok=False)
    import catboost
    if args.package is not None:
        assert Path(catboost.__file__).resolve().is_relative_to(args.package), 'selected package was not imported'
    original = catboost.CatBoost._fit
    def require_gpu(self, *values, **kwargs):
        assert self.get_params().get('task_type') == 'GPU', 'CPU training is forbidden in release checks'
        return original(self, *values, **kwargs)
    catboost.CatBoost._fit = require_gpu
    report = dict(command=args.command, status='running', cases=[],
                  python=sys.executable, package=str(Path(catboost.__file__).resolve()),
                  extension_sha256=sha256(Path(catboost.__file__).parent / '_catboost.so'),
                  source_sha256=sha256(__file__), started_unix=time.time())
    try:
        if args.command == 'legacy-replay':
            from legacy import replay
            replay(args, report)
        else:
            globals()[args.command.replace('-', '_')](args, report)
        report['status'] = 'passed'
    except BaseException:
        report['status'] = 'failed'
        report['error'] = traceback.format_exc()
        raise
    finally:
        report['elapsed_seconds'] = time.time() - report['started_unix']
        (args.output_dir / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
        print(f"{report['status']}: {len(report['cases'])} cases; {args.output_dir / 'report.json'}", flush=True)
        catboost.CatBoost._fit = original


if __name__ == '__main__':
    main()
