"""List or execute native/CLI training-mode release smoke configurations.

Custom MSL objectives require a native Python descriptor and are covered by
the native acceptance tests, rather than this CLI-compatible grid.
"""
import argparse
from collections import Counter
from itertools import product
import json
from pathlib import Path
import subprocess
import sys
import time
import traceback

from run import load_helper, sha256


QUERY_LOSSES = ('QueryRMSE', 'QuerySoftMax:beta=0.7;lambda=0.03', 'PairLogit',
                'YetiRank:permutations=7;decay=0.85')
COMBINATIONS = (
    ('scalar', 'Combination:loss0=Huber:delta=0.8;weight0=0.4;loss1=RMSE;weight1=0.6'),
    ('query', 'Combination:loss0=RMSE;weight0=0.4;loss1=QueryRMSE;weight1=0.6'),
)


def configurations():
    for policy in ('Depthwise', 'Lossguide', 'Region'):
        for category, count in (('numeric', 1), ('onehot', 1), ('ctr', 1), ('ctr', 4)):
            yield dict(name=f'greedy-{policy}-{category}-p{count}', family='greedy-yeti',
                       policy=policy, boosting='Plain', category=category, count=count,
                       loss='YetiRank:permutations=7;decay=0.85;mode=Classic')
    for loss, boosting, category, count in product(QUERY_LOSSES, ('Plain', 'Ordered'),
                                                  ('numeric', 'onehot', 'ctr', 'compound'), (1, 4)):
        yield dict(name=f'fp-{boosting}-{loss.partition(":")[0]}-{category}-p{count}', family='fp-query',
                   policy='SymmetricTree', boosting=boosting, category=category, count=count, loss=loss)
    for (label, loss), boosting, category, count in product(COMBINATIONS, ('Plain', 'Ordered'),
                                                          ('numeric', 'ctr', 'compound'), (1, 4)):
        yield dict(name=f'combination-{label}-{boosting}-{category}-p{count}', family='combination-' + label,
                   policy='SymmetricTree', boosting=boosting, category=category, count=count, loss=loss)


def options(case):
    result = dict(task_type='GPU', loss_function=case['loss'], boosting_type=case['boosting'],
                  grow_policy=case['policy'], iterations=7, depth=3, learning_rate=.15,
                  random_seed=713, border_count=16, bootstrap_type='No', random_strength=0,
                  score_function='Cosine', l2_leaf_reg=2, leaf_estimation_method='Newton',
                  leaf_estimation_iterations=3, leaf_estimation_backtracking='No',
                  permutation_count=case['count'], has_time=case['count'] == 1,
                  metric_period=1, logging_level='Silent', allow_writing_files=False,
                  one_hot_max_size=8 if case['category'] == 'onehot' else 1,
                  max_ctr_complexity=2 if case['category'] == 'compound' else 1)
    if case['family'] == 'greedy-yeti':
        result.update(iterations=5, learning_rate=.13, random_seed=817, score_function='NewtonL2',
                      eval_metric='PFound:top=5;decay=0.7')
        if case['policy'] == 'Lossguide':
            result['max_leaves'] = 6
    else:
        result['data_partition'] = 'FeatureParallel'
    if case['category'] in ('ctr', 'compound'):
        ctr = 'Borders:CtrBorderType=Uniform:CtrBorderCount=15:Prior=0.5'
        result.update(simple_ctr=[ctr], combinations_ctr=[ctr], ctr_target_border_count=1,
                      ctr_history_unit='Group', counter_calc_method='SkipTest', model_size_reg=0)
    if case['category'] == 'compound':
        result.update(iterations=8, depth=4)
    if case['boosting'] == 'Ordered':
        result.update(min_fold_size=8, fold_len_multiplier=1.7, fold_permutation_block=3)
    if case['family'].startswith('combination'):
        result['boost_from_average'] = False
    return result


def literal_pairs(y, groups):
    import numpy as np
    edges = []
    for group in np.unique(groups):
        rows = np.flatnonzero(groups == group)
        ranked = rows[np.argsort(y[rows], kind='stable')]
        edges.extend((int(a), int(b)) for a, b in zip(ranked[1:], ranked[:-1]) if y[a] > y[b])
    edges = np.asarray(edges, np.uint32).reshape(-1, 2)
    assert len(edges)
    weights = (.25 + np.arange(len(edges)) % 11 / 7).astype(np.float32)
    weights[::13] = 0
    return edges, weights


def data(case):
    import numpy as np
    if case['category'] == 'numeric':
        rng = np.random.default_rng(4812)
        groups = np.repeat(np.arange(24, dtype=np.uint64), np.tile([3, 5, 7, 4, 6, 8], 4))
        x = rng.normal(size=(len(groups), 4)).astype(np.float32)
        signal = 1.3 * x[:, 0] - .6 * x[:, 1] + .4 * x[:, 0] * x[:, 2]
        y = (signal + (groups % 5) * .75).astype(np.float32) if case['loss'] == 'QueryRMSE' else (
            (signal > 0).astype(np.float32) if case['family'].startswith('combination') else
            (.125 + .75 / (1 + np.exp(-signal))).astype(np.float32))
        cats = []
        weights = (.4 + np.arange(len(x)) % 9 / 5).astype(np.float32)
        weights[::17] = 0
    else:
        helper, _ = load_helper('smoke.py')
        x, y, _, weights = helper.data(2)
        if not case['family'].startswith('combination'):
            y = (.125 + .75 * y).astype(np.float32)
        cats = [0, 1]
        groups = np.arange(len(x), dtype=np.uint64) // 6
    future = x.copy()
    if cats:
        future[::17, cats[0]] = 'never-seen'
    else:
        future[:, 0] += np.float32(.1)
    pool_options = dict(cat_features=cats, group_id=groups, weight=weights)
    if case['loss'] == 'PairLogit':
        pool_options['pairs'], pool_options['pairs_weight'] = literal_pairs(y, groups)
    return x, y, future, pool_options


def readers(model, future, folder, case):
    import numpy as np
    from catboost import CatBoost
    raw = dict(prediction_type='RawFormulaVal')
    expected = model.predict(future, task_type='GPU', **raw)
    assert np.isfinite(expected).all()
    np.testing.assert_allclose(model.predict(future, **raw), expected, rtol=8e-6, atol=2e-6)
    for fmt in ('cbm', 'json'):
        path = folder / ('export.' + fmt)
        model.save_model(path, format=fmt)
        restored = CatBoost().load_model(path, format=fmt)
        assert restored.get_metadata()['metal_backend'] == 'METAL'
        np.testing.assert_allclose(restored.predict(future, task_type='GPU', **raw), expected, rtol=8e-6, atol=2e-6)
        np.testing.assert_allclose(restored.predict(future, **raw), expected, rtol=8e-6, atol=2e-6)
    document = json.loads((folder / 'export.json').read_text())
    compound = sum(len(ctr['elements']) > 1 for ctr in document.get('features_info', {}).get('ctrs', []))
    if case['category'] == 'compound':
        assert compound > 0, 'model did not select a compound CTR: ' + case['name']
    np.savez(folder / 'predictions.npz', gpu_predictions=expected,
             leaf_values=model.get_leaf_values(), leaf_weights=model.get_leaf_weights())
    return compound


def execute(args, report, grid):
    from catboost import CatBoost, Pool
    if args.command == 'cli':
        if args.cli is None or not args.cli.is_file():
            raise ValueError('--cli must name the frozen executable')
        report.update(cli=str(args.cli), cli_sha256=sha256(args.cli))
    for case in grid:
        folder = args.output_dir / case['name']
        folder.mkdir()
        x, y, future, pool_options = data(case)
        config = options(case)
        (folder / 'params.json').write_text(json.dumps(config, indent=2) + '\n')
        if args.command == 'smoke':
            pool = Pool(x, y, **pool_options)
            model = CatBoost().set_params(**config).fit(pool, eval_set=pool, use_best_model=False)
        else:
            learn, cd, cbm = (folder / name for name in ('learn.tsv', 'columns.cd', 'model.cbm'))
            learn.write_text(''.join('\t'.join(map(str, (y[i], pool_options['weight'][i],
                              pool_options['group_id'][i], *x[i]))) + '\n' for i in range(len(y))))
            cd.write_text('0\tLabel\n1\tWeight\n2\tGroupId\n' + ''.join(
                f'{i + 3}\t' + ('Categ' if i in pool_options['cat_features'] else 'Num') + f'\tfeature{i}\n'
                for i in range(x.shape[1])))
            command = [str(args.cli), 'fit', '--task-type', 'GPU', '--params-file', str(folder / 'params.json'),
                       '--learn-set', str(learn), '--test-set', str(learn), '--column-description', str(cd),
                       '--use-best-model', 'false', '--allow-writing-files', 'false', '--model-file', str(cbm)]
            if 'pairs' in pool_options:
                pairs = folder / 'pairs.tsv'
                pairs.write_text(''.join(f'{winner}\t{loser}\t{weight}\n' for (winner, loser), weight in
                                        zip(pool_options['pairs'], pool_options['pairs_weight'])))
                command += ['--learn-pairs', str(pairs), '--test-pairs', str(pairs)]
            (folder / 'command.json').write_text(json.dumps(command, indent=2) + '\n')
            with (folder / 'fit.log').open('w') as output:
                subprocess.run(command, stdout=output, stderr=subprocess.STDOUT, check=True)
            model = CatBoost().load_model(cbm)
        assert model.get_metadata()['metal_backend'] == 'METAL'
        assert model.tree_count_ == config['iterations']
        compound = readers(model, future, folder, case)
        report['cases'].append(case | dict(trees=model.tree_count_, compound_ctrs=compound,
                              actual_permutations=model.get_metadata()['metal_permutations'],
                              parameters_sha256=sha256(folder / 'params.json')))
        print('PASS ' + args.command + ' native Metal: ' + case['name'], flush=True)
    assert len(report['cases']) == report['expected_cases']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('list', 'smoke', 'cli'))
    parser.add_argument('--family', action='append', choices=('greedy-yeti', 'fp-query', 'combination-scalar', 'combination-query'))
    parser.add_argument('--package', type=Path)
    parser.add_argument('--cli', type=Path)
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args()
    grid = [case for case in configurations() if not args.family or case['family'] in args.family]
    assert len({case['name'] for case in grid}) == len(grid)
    inventory = dict(expected_cases=len(grid), families=dict(Counter(case['family'] for case in grid)), cases=grid)
    if args.command == 'list':
        print(json.dumps(inventory, indent=2))
        return
    if args.output_dir is None:
        parser.error('--output-dir is required for execution')
    for key in ('package', 'cli', 'output_dir'):
        value = getattr(args, key)
        if value is not None:
            setattr(args, key, value.absolute())
    if args.package is not None:
        sys.path.insert(0, str(args.package))
    import catboost
    if args.package is not None:
        assert Path(catboost.__file__).resolve().is_relative_to(args.package.resolve())
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / 'inventory.json').write_text(json.dumps(inventory, indent=2) + '\n')
    original = catboost.CatBoost._fit
    def checked(self, *values, **kwargs):
        assert args.command == 'smoke', 'CLI checks must not fit through Python'
        assert self.get_params().get('task_type') == 'GPU'
        return original(self, *values, **kwargs)
    catboost.CatBoost._fit = checked
    fixture_path = Path(__file__).resolve().parents[1] / 'compound_ctr_acceptance/smoke.py'
    report = dict(command=args.command, status='running', cases=[], python=sys.executable,
                  expected_cases=len(grid), expected_families=inventory['families'],
                  package=str(Path(catboost.__file__).resolve()), source_sha256=sha256(__file__),
                  fixture_helper=str(fixture_path), fixture_helper_sha256=sha256(fixture_path),
                  extension_sha256=sha256(Path(catboost.__file__).parent / '_catboost.so'), started_unix=time.time())
    try:
        execute(args, report, grid)
        assert sha256(fixture_path) == report['fixture_helper_sha256'], 'fixture helper changed during acceptance'
        report['status'] = 'passed'
    except BaseException:
        report.update(status='failed', error=traceback.format_exc())
        raise
    finally:
        report['elapsed_seconds'] = time.time() - report['started_unix']
        (args.output_dir / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
        print(f"{report['status']}: {len(report['cases'])}/{len(grid)} cases; {args.output_dir / 'report.json'}", flush=True)
        catboost.CatBoost._fit = original


if __name__ == '__main__':
    main()
