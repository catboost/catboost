"""Replay the 194 immutable pre-PairLogit snapshots saved in 155047Z.

The data generators and options below reproduce the two saved verification
programs, without executing their orchestration or importing mutable tests.
The original snapshots were generated using checkpoint 151643Z, then verified
by 155047Z. Only necessary fixture files are copied into new output folders.
"""

import json
from pathlib import Path
import shutil

import numpy as np
from catboost import CatBoostClassifier, CatBoostRanker, CatBoostRegressor, Pool
from catboost_metal import CatBoostMetalClassifier, CatBoostMetalRanker, CatBoostMetalRegressor

from run import ResumedIterations, assert_exact, native_arrays, sha256


def ordered_cases():
    for api in ('native', 'standalone'):
        for loss in ('RMSE', 'Logloss', 'Huber:delta=1.2'):
            for count in (1, 4):
                for bootstrap in ('No', 'MVS'):
                    for category in ('numeric', 'onehot', 'grouped_numeric', 'grouped_onehot'):
                        rng = np.random.default_rng(72183)
                        rows = 192
                        numeric = rng.integers(0, 4, (2, rows), dtype=np.uint8)
                        code = rng.integers(0, 4, rows)
                        other = rng.integers(0, 3, rows)
                        x = np.empty((rows, 4), object)
                        x[:, 0], x[:, 2] = numeric[0].astype(float), numeric[1].astype(float)
                        x[:, 1] = np.array(['amber', 'blue', 'cyan', '黑'])[code]
                        x[:, 3] = np.array(['a', 'b', 'c'])[other]
                        y = np.float32(2.1 * (code == 1) - 1.3 * (code == 2) + .8 * (other == 0) + .15 * numeric[0] - .1 * numeric[1] - .4)
                        if loss == 'Logloss':
                            y = (y > 0).astype(np.float32)
                        weights = rng.uniform(.3, 2, rows).astype(np.float32)
                        weights[::19] = 0
                        config = dict(task_type='GPU', boosting_type='Ordered', grow_policy='SymmetricTree',
                                      iterations=5, depth=3, learning_rate=.15, random_seed=47, border_count=20,
                                      bootstrap_type=bootstrap, random_strength=.4, score_function='Cosine',
                                      leaf_estimation_backtracking='No', boost_from_average=False, metric_period=1,
                                      verbose=False, allow_writing_files=False, loss_function=loss,
                                      leaf_estimation_method='Newton', leaf_estimation_iterations=3,
                                      one_hot_max_size=5, has_time=False, permutation_count=count,
                                      min_fold_size=16, fold_len_multiplier=1.7, fold_permutation_block=3)
                        group = {}
                        if category.startswith('grouped'):
                            sizes = [1, 3, 37, 2, 61, 7, 5, 76]
                            group = dict(group_id=np.repeat(np.arange(len(sizes), dtype=np.uint64) + (1 << 40), sizes),
                                         group_weight=np.repeat(np.linspace(.5, 1.75, len(sizes), dtype=np.float32), sizes))
                        cats = [1, 3]
                        if category.endswith('numeric'):
                            x, cats = x[:, [0, 2]].astype(np.float32), []
                        name = f'{api}-{loss.split(":")[0]}-p{count}-{bootstrap}-{category}'
                        yield dict(name=name, api=api, loss=loss, x=x, y=y, weights=weights,
                                   config=config, group=group, cats=cats, ranking=False, ordered=True)


def extra_cases():
    rng = np.random.default_rng(1391)
    all_x = rng.normal(size=(120, 3)).astype(object)
    all_x[:, 2] = np.array(['cat' + str(i) for i in range(6)])[np.arange(120) % 6]
    raw = np.asarray(all_x[:, :2], float)
    signal = raw[:, 0] + .8 * raw[:, 1]
    weights = np.linspace(.3, 2.1, 120, dtype=np.float32)
    for api in ('native', 'standalone'):
        for loss in ('RMSE', 'Logloss', 'MultiClass', 'MultiClassOneVsAll', 'RMSEWithUncertainty', 'QueryRMSE', 'QuerySoftMax'):
            ranking = loss.startswith('Query')
            classification = loss in ('Logloss', 'MultiClass', 'MultiClassOneVsAll')
            policies = ('SymmetricTree',) if ranking else ('Depthwise', 'Lossguide', 'Region')
            for policy in policies:
                for category in (('numeric', 'onehot') if ranking else ('numeric', 'onehot', 'ctr')):
                    x = all_x[:, :2].astype(np.float32) if category == 'numeric' else all_x.copy()
                    cats = [] if category == 'numeric' else [2]
                    y = (np.exp(signal / 3).astype(np.float32) if loss == 'QuerySoftMax' else
                         (signal > 0).astype(np.float32) if loss == 'Logloss' else
                         np.digitize(signal, [-.4, .5]).astype(np.float32) if classification else signal.astype(np.float32))
                    config = dict(loss_function=loss, iterations=5, depth=3, grow_policy=policy, learning_rate=.2,
                                  l2_leaf_reg=2., score_function='Cosine',
                                  leaf_estimation_method='Gradient' if loss == 'QuerySoftMax' else 'Newton',
                                  leaf_estimation_iterations=3, leaf_estimation_backtracking='Armijo', random_seed=45,
                                  random_strength=.2, bootstrap_type='Bernoulli', subsample=.8, border_count=8,
                                  one_hot_max_size=8 if category == 'onehot' else 1)
                    if policy == 'Lossguide':
                        config['max_leaves'] = 6
                    if api == 'native':
                        config.update(task_type='GPU', allow_writing_files=False, verbose=False)
                        if category == 'ctr':
                            config.update(max_ctr_complexity=1, permutation_count=4, ctr_target_border_count=1,
                                          simple_ctr=['Borders:CtrBorderCount=7:TargetBorderCount=1:Prior=0.5'],
                                          counter_calc_method='SkipTest')
                    elif category == 'ctr':
                        config['permutation_count'] = 4
                    group = dict(group_id=np.repeat(np.arange(20), 6)) if ranking else {}
                    yield dict(name=f'{api}-{loss}-{policy}-{category}', api=api, loss=loss, x=x, y=y,
                               weights=weights, config=config, group=group, cats=cats, ranking=ranking, ordered=False)


def standalone_arrays(model, case, folder):
    result = model._result
    assert result.stats['resumed_iterations'] == 2
    if case['ordered']:
        with np.load(folder / 'full.npz', allow_pickle=False) as full, np.load(folder / 'resume.npz', allow_pickle=False) as resumed:
            for key in full.files:
                if key.startswith('ordered_'):
                    np.testing.assert_array_equal(full[key], resumed[key], err_msg=case['name'] + ':' + key)
        arrays = {key: getattr(result, key) for key in ('depths', 'split_features', 'split_bins', 'split_types', 'leaf_values', 'leaf_weights', 'predictions', 'rmse')}
    else:
        arrays = dict(predictions=model.training_predictions_, loss=model.loss_history_)
        if case['ranking']:
            for key in ('depths', 'split_features', 'split_bins', 'leaf_values', 'leaf_weights'):
                arrays[key] = getattr(result, key)
        else:
            for index, tree in enumerate(result.trees):
                for key in ('nodes', 'leaf_values', 'leaf_weights'):
                    arrays[f't{index}-{key}'] = getattr(tree, key)
    options = {} if case['ranking'] else dict(prediction_type='RawFormulaVal')
    arrays['gpu_predictions'] = model.predict(case['x'], task_type='GPU', **options)
    arrays['history'] = np.array(json.dumps(model.get_evals_result(), sort_keys=True))
    assert np.isfinite(arrays['gpu_predictions']).all()
    return arrays


def replay_case(case, source, folder):
    x, y, weights, group = (case[key] for key in ('x', 'y', 'weights', 'group'))
    config = case['config'].copy()
    classification = case['loss'] in ('Logloss', 'MultiClass', 'MultiClassOneVsAll')
    if case['api'] == 'native':
        cls = CatBoostRanker if case['ranking'] else CatBoostClassifier if classification else CatBoostRegressor
        pool = Pool(x, y, weight=weights, cat_features=case['cats'], **{key: value for key, value in group.items() if key != 'group_weight'})
        if 'group_weight' in group:
            pool.set_group_weight(group['group_weight'])
        config.update(save_snapshot=True, snapshot_interval=0, snapshot_file='resume.snapshot' if case['ordered'] else 'state.snapshot',
                      allow_writing_files=True, train_dir=str(folder))
        resumed = ResumedIterations()
        model = cls().set_params(**config).fit(pool, eval_set=pool, use_best_model=False, callbacks=[resumed])
        resumed.verify()
        arrays = native_arrays(model, x)
    else:
        cls = CatBoostMetalRanker if case['ranking'] else CatBoostMetalClassifier if classification else CatBoostMetalRegressor
        for key in ('task_type', 'verbose', 'allow_writing_files', 'has_time', 'metric_period'):
            config.pop(key, None)
        config['cat_features'] = case['cats']
        fit = dict(sample_weight=weights, eval_set=(x, y, group['group_id'], weights) if case['ranking'] else (x, y, weights),
                   use_best_model=False, **group)
        fit.update(save_snapshot=True, snapshot_interval=0, snapshot_file=folder / ('resume.npz' if case['ordered'] else 'state.npz'))
        model = cls(**config).fit(x, y, **fit)
        arrays = standalone_arrays(model, case, folder)
    assert model.tree_count_ == 5
    assert_exact(arrays, source / 'expected.npz', case['name'])


def replay(args, report):
    import catboost_metal
    report.update(snapshot_generated_by='20260913T151643Z', snapshot_previously_validated_by='20260913T155047Z',
                  snapshot_archive=str(args.release), standalone_package=str(Path(catboost_metal.__file__).resolve()),
                  legacy_runner_sha256=sha256(__file__))
    suites = [('catbooster-greedy-query-legacy', list(ordered_cases())),
              ('catbooster-greedy-query-legacy-extra', list(extra_cases()))]
    assert [len(cases) for _, cases in suites] == [96, 98]
    # Validate the complete inventory before the first resumed fit.
    fixture_hashes = {}
    for suite, cases in suites:
        archive = args.release / suite / 'original-snapshots'
        assert {p.name for p in archive.iterdir() if p.is_dir()} == {case['name'] for case in cases}
        for case in cases:
            source = archive / case['name']
            suffix = '.snapshot' if case['api'] == 'native' else '.npz'
            files = ['expected.npz', ('resume' if case['ordered'] else 'state') + suffix]
            if case['ordered'] and case['api'] == 'standalone':
                files.append('full.npz')
            for name in files:
                path = source / name
                fixture_hashes[str(path)] = sha256(path)
            case['fixture_files'] = files
    report['source_fixture_sha256'] = fixture_hashes
    for suite, cases in suites:
        for case in cases:
            source = args.release / suite / 'original-snapshots' / case['name']
            folder = args.output_dir / suite / case['name']
            folder.mkdir(parents=True)
            for name in case['fixture_files']:
                shutil.copy2(source / name, folder / name)
            replay_case(case, source, folder)
            report['cases'].append(dict(suite=suite, name=case['name'], exact=True))
            print('PASS preserved snapshot exact:', case['name'], flush=True)
    assert all(sha256(path) == value for path, value in fixture_hashes.items()), 'source fixtures changed during replay'
