"""CLI and installed-wheel compound CTR smoke checks with standard readers."""

import argparse
from itertools import product
import json
from pathlib import Path
import subprocess
import sys
import time
import traceback

from run import sha256


def data(complexity):
    """Unequal joint frequencies expose a stronger categorical interaction."""
    import numpy as np
    rows, labels = [], []
    cardinality = 6 if complexity == 2 else 4
    for values in product(range(cardinality), repeat=complexity):
        if values == (0,) * (complexity - 1) + (1,):
            continue
        parity = sum(values) % 2
        count = 8 + 2 * values[0] + 10 * parity
        if complexity == 3:
            count += 4 * ((values[0] + values[1]) % 2)
        for _ in range(count):
            rows.append([f'cat-{column}-{value}' for column, value in enumerate(values)])
            labels.append(parity)
    order = np.random.default_rng(831).permutation(len(rows))
    x = np.asarray(rows, dtype=object)[order]
    y = np.asarray(labels, dtype=np.float32)[order]
    future = [[f'cat-{column}-{value}' for column, value in enumerate(values)]
              for values in product(range(cardinality), repeat=complexity)]
    future.extend([['never-seen'] * complexity,
                   ['never-seen'] + [f'cat-{column}-0' for column in range(1, complexity)]])
    weights = (.4 + (np.arange(len(x)) % 9) / 5).astype(np.float32)
    weights[::29] = 0
    return x, y, np.asarray(future, dtype=object), weights


def config(boosting, count, complexity, loss):
    ctr = 'Borders:CtrBorderType=Uniform:CtrBorderCount=15:Prior=0.5'
    result = dict(task_type='GPU', data_partition='FeatureParallel', boosting_type=boosting,
                  grow_policy='SymmetricTree', loss_function=loss,
                  iterations=8 if complexity == 2 else 12, depth=4 if complexity == 2 else 5,
                  learning_rate=.2, l2_leaf_reg=2, random_seed=619, bootstrap_type='No',
                  random_strength=0, score_function='Cosine', leaf_estimation_method='Newton',
                  leaf_estimation_iterations=2, leaf_estimation_backtracking='No',
                  boost_from_average=False, metric_period=1, logging_level='Silent',
                  allow_writing_files=False, one_hot_max_size=2, max_ctr_complexity=complexity,
                  model_size_reg=0, simple_ctr=[ctr], combinations_ctr=[ctr],
                  ctr_target_border_count=1, ctr_history_unit='Sample', counter_calc_method='SkipTest',
                  border_count=16, permutation_count=count, has_time=count == 1)
    if boosting == 'Ordered':
        result.update(min_fold_size=16, fold_len_multiplier=1.7, fold_permutation_block=3)
    return result


def readers(model, cls, x, folder, complexity):
    import numpy as np
    raw = dict(prediction_type='RawFormulaVal')
    expected = model.predict(x, task_type='GPU', **raw)
    assert np.isfinite(expected).all()
    np.testing.assert_allclose(model.predict(x, **raw), expected, rtol=5e-6, atol=7e-7)
    for fmt in ('cbm', 'json'):
        path = folder / ('export.' + fmt)
        model.save_model(path, format=fmt)
        restored = cls().load_model(path, format=fmt)
        assert restored.get_metadata()['metal_backend'] == 'METAL'
        np.testing.assert_allclose(restored.predict(x, task_type='GPU', **raw), expected, rtol=5e-6, atol=7e-7)
        np.testing.assert_allclose(restored.predict(x, **raw), expected, rtol=5e-6, atol=7e-7)
    document = json.loads((folder / 'export.json').read_text())
    projections = [ctr['elements'] for ctr in document['features_info'].get('ctrs', [])]
    compound = [elements for elements in projections if len(elements) > 1]
    assert compound, 'model did not select a compound CTR'
    assert all(len(elements) <= complexity for elements in projections)
    np.savez(folder / 'predictions.npz', predictions=expected)
    return len(compound), max(map(len, projections))


def execute(args, report):
    from catboost import CatBoostClassifier, CatBoostRegressor, Pool
    if args.command == 'cli':
        if args.cli is None or not args.cli.is_file():
            raise ValueError('--cli must name the frozen executable')
        report.update(cli=str(args.cli), cli_sha256=sha256(args.cli))
    for boosting, count, complexity, loss in product(('Plain', 'Ordered'), (1, 4), (2, 3), ('RMSE', 'Logloss')):
        name = f'{boosting}-p{count}-complexity{complexity}-{loss}'
        folder = args.output_dir / name
        folder.mkdir()
        x, y, future, weights = data(complexity)
        options = config(boosting, count, complexity, loss)
        (folder / 'params.json').write_text(json.dumps(options, indent=2) + '\n')
        cls = CatBoostClassifier if loss == 'Logloss' else CatBoostRegressor
        if args.command == 'smoke':
            pool = Pool(x, y, weight=weights, cat_features=list(range(complexity)))
            model = cls().set_params(**options).fit(pool, eval_set=pool, use_best_model=False)
        else:
            learn, cd, cbm = (folder / file for file in ('learn.tsv', 'columns.cd', 'model.cbm'))
            learn.write_text(''.join('\t'.join(map(str, (y[i], weights[i], *x[i]))) + '\n' for i in range(len(y))))
            cd.write_text('0\tLabel\n1\tWeight\n' + ''.join(f'{i + 2}\tCateg\tfeature{i}\n' for i in range(complexity)))
            command = [str(args.cli), 'fit', '--task-type', 'GPU', '--params-file', str(folder / 'params.json'),
                       '--learn-set', str(learn), '--test-set', str(learn), '--column-description', str(cd),
                       '--use-best-model', 'false', '--allow-writing-files', 'false', '--model-file', str(cbm)]
            (folder / 'command.json').write_text(json.dumps(command, indent=2) + '\n')
            with (folder / 'fit.log').open('w') as output:
                subprocess.run(command, stdout=output, stderr=subprocess.STDOUT, check=True)
            model = cls().load_model(cbm)
        assert model.get_metadata()['metal_backend'] == 'METAL'
        assert model.get_metadata()['metal_permutations'] == str(count)
        assert model.tree_count_ == options['iterations']
        selected, largest = readers(model, cls, future, folder, complexity)
        report['cases'].append(dict(name=name, trees=model.tree_count_, compound_ctrs=selected,
                                   largest_projection=largest, parameters_sha256=sha256(folder / 'params.json')))
        print(f'PASS {args.command} compound GPU: {name}', flush=True)
    assert len(report['cases']) == 16


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('smoke', 'cli'))
    parser.add_argument('--package', type=Path)
    parser.add_argument('--cli', type=Path)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
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
    original_fit = catboost.CatBoost._fit
    def checked_fit(self, *values, **kwargs):
        assert args.command == 'smoke', 'CLI checks must not train through Python'
        assert self.get_params().get('task_type') == 'GPU', 'CPU training is forbidden'
        return original_fit(self, *values, **kwargs)
    catboost.CatBoost._fit = checked_fit
    report = dict(command=args.command, status='running', cases=[], python=sys.executable,
                  package=str(Path(catboost.__file__).resolve()), source_sha256=sha256(__file__),
                  extension_sha256=sha256(Path(catboost.__file__).parent / '_catboost.so'), started_unix=time.time())
    try:
        execute(args, report)
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
