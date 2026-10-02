"""Independent fixed-topology leaves and exact Plain greedy dataset recovery."""
import ctypes as ct
import numpy as np
import pytest
from catboost_metal import _greedy
from catboost_metal._data import cuda_search_permutation
from test_greedy_objectives import CASES, problem
from test_greedy_training import POLICIES, route
from test_backtracking import cuda_leaf_walker
from cuda_scalar_reference import exact_leaf_value


@pytest.fixture(autouse=True)
def forbid_cpu_fit(monkeypatch):
    from catboost import CatBoost
    def forbidden(*args, **kwargs): raise AssertionError('CPU CatBoost fitting is forbidden')
    monkeypatch.setattr(CatBoost, '_fit', forbidden)


def inputs(policy='Depthwise', case=('RMSE', None, 'Newton'), kind='No', count=4):
    loss, param, method = case
    bins, y, w, cf, cb = problem(loss, rows=137)
    # Include equality predicates and an empty leaf in some alternate banks.
    rng = np.random.default_rng(83541)
    banks = np.array([bins, *[rng.integers(0,7,bins.shape,dtype=np.uint8) for _ in range(count-1)]])
    if count > 1: banks[-1, 1] = 0
    types = np.uint8(cf == 2)
    cursors = np.array([np.linspace(-.1,.2,len(y),dtype=np.float32) + np.float32(p*.03) for p in range(count)])
    args = dict(bins=bins, targets=y, candidate_features=cf, candidate_bins=cb,
        candidate_types=types, sample_weight=w, initial_predictions=cursors[0],
        grow_policy=policy, objective=loss, objective_param=param, iterations=4, depth=3,
        max_leaves=7 if policy!='Region' else 4, learning_rate=.19, l2_leaf_reg=1.7,
        leaf_estimation_method=method, leaf_estimation_iterations=3,
        score_function='L2', bootstrap_type=kind, subsample=.71, random_seed=61873)
    return args, banks, cursors


def fixed(tree, bins, args, cursor):
    ids = route(tree, bins); leaves = len(tree.leaf_values)
    if args['leaf_estimation_method'] == 'Exact':
        w, y = args['sample_weight'], args['targets']
        values = np.array([exact_leaf_value(np.float32(y-cursor)[ids==leaf], w[ids==leaf],
            args['objective'], args['objective_param'] or .5, targets=y[ids==leaf])
            for leaf in range(leaves)], np.float32)
        weights = np.bincount(ids, weights=w.astype(float), minlength=leaves)
    else:
        values, weights, _ = cuda_leaf_walker(args['targets'], cursor, args['sample_weight'], ids, leaves,
            **{k:args[k] for k in ('objective','objective_param','l2_leaf_reg',
                'leaf_estimation_method','leaf_estimation_iterations')},
            leaf_estimation_backtracking=args.get('leaf_estimation_backtracking','No'))
    return np.float32(values * np.float32(args['learning_rate'])), weights, ids


def compare_walk(args, banks, cursors, schedule='cuda'):
    with _greedy.Session(**args) as session:
        session.configure_permutations(banks, cursors)
        for iteration in range(args['iterations']):
            selected = cuda_search_permutation(args['random_seed'], iteration, len(banks)) if schedule=='cuda' else [3,2,0,1][iteration]
            with _greedy.Session(**(args | dict(bins=banks[selected], initial_predictions=cursors[selected],
                    iterations=1, iteration_offset=iteration))) as single:
                searched = single.step()
            session.select_permutation(selected); actual = session.step()
            np.testing.assert_array_equal(actual.nodes, searched.nodes)
            for p, bank in enumerate(banks):
                values, weights, ids = fixed(searched, bank, args, cursors[p])
                if p == selected:
                    np.testing.assert_allclose(searched.leaf_values, values, rtol=3e-4, atol=3e-6)
                    values = searched.leaf_values
                cursors[p] = np.float32(cursors[p] + values[ids])
                if p == len(banks)-1:
                    np.testing.assert_allclose(actual.leaf_values, values, rtol=3e-4, atol=3e-6)
                    np.testing.assert_allclose(actual.leaf_weights, weights, rtol=4e-6, atol=4e-6)
            state = session.permutation_state
            np.testing.assert_allclose(state['predictions'], cursors, rtol=3e-4, atol=4e-6)
            np.testing.assert_array_equal(session.predictions(), state['predictions'][-1])
            np.testing.assert_array_equal(state['mvs_lambdas'], 0)
            np.testing.assert_array_equal(state['mvs_valid'], 0)
            # Subsequent structure checks use the exact resident cursors;
            # independent equations above check every per-dataset update.
            cursors = state['predictions']


@pytest.mark.parametrize('case', CASES)
@pytest.mark.parametrize('policy', POLICIES)
@pytest.mark.parametrize('schedule', ['cuda','all'])
def test_each_dataset_estimates_fixed_structure_with_original_weights(case,policy,schedule):
    compare_walk(*inputs(policy,case,kind='Bernoulli'), schedule)


@pytest.mark.parametrize('policy', POLICIES)
@pytest.mark.parametrize('mode', ['No','AnyImprovement','Armijo'])
@pytest.mark.parametrize('method', ['Newton','Gradient'])
def test_binary_backtracking_is_independent_per_cursor(policy,mode,method):
    args,banks,cursors=inputs(policy,('Logloss',None,method),'Bayesian')
    args['leaf_estimation_backtracking']=mode
    compare_walk(args,banks,cursors,'all')


@pytest.mark.parametrize('policy', POLICIES)
@pytest.mark.parametrize('loss', ['Quantile','MAE','MAPE'])
def test_exact_leaves_use_each_dataset_residual_quantiles(policy,loss):
    args,banks,cursors=inputs(policy,(loss,.31 if loss=='Quantile' else None,'Exact'),'Poisson')
    compare_walk(args,banks,cursors,'all')


@pytest.mark.parametrize('policy', POLICIES)
@pytest.mark.parametrize('kind', ['No','Bayesian','Bernoulli','Poisson'])
def test_all_cursors_resume_exactly_at_absolute_iteration(policy,kind):
    args,banks,cursors=inputs(policy,kind=kind)
    args.update(score_function='Cosine',random_strength=.3)
    with _greedy.Session(**args) as direct:
        direct.configure_permutations(banks,cursors)
        for t in range(2):
            direct.select_permutation(cuda_search_permutation(args['random_seed'],t,4));direct.step()
        state=direct.permutation_state
        with _greedy.Session(**(args|dict(iterations=2,iteration_offset=2))) as resumed:
            resumed.configure_permutations(banks,state['predictions'])
            for t in range(2,4):
                selected=cuda_search_permutation(args['random_seed'],t,4)
                direct.select_permutation(selected);resumed.select_permutation(selected)
                a,b=direct.step(),resumed.step()
                for key in ('nodes','leaf_values','leaf_weights'):
                    np.testing.assert_array_equal(getattr(a,key),getattr(b,key))
                np.testing.assert_array_equal(direct.permutation_state['predictions'],resumed.permutation_state['predictions'])
                assert a.loss==b.loss


@pytest.mark.parametrize('policy', POLICIES)
@pytest.mark.parametrize('count', [1,2,4,7,64])
def test_identical_banks_preserve_p1_forests_bitwise(policy,count):
    args,_,_=inputs(policy,count=count);args['iterations']=2
    banks=np.repeat(args['bins'][None],count,axis=0)
    with _greedy.Session(**args) as single, _greedy.Session(**args) as many:
        many.configure_permutations(banks)
        for _ in range(2):
            many.select_permutation(count-1)
            a,b=single.step(),many.step()
            for key in ('nodes','leaf_values','leaf_weights'):np.testing.assert_array_equal(getattr(a,key),getattr(b,key))
            np.testing.assert_array_equal(many.permutation_state['predictions'],np.repeat(single.predictions()[None],count,axis=0))
            assert a.loss==b.loss


@pytest.mark.parametrize('bad', ['shape','range','count','nan','cursor_shape','repeat','after','index','mvs','capacity'])
def test_invalid_permutation_geometry_and_state_are_rejected(bad):
    args,banks,cursors=inputs()
    with _greedy.Session(**args) as session:
        if bad in ('shape','range','count','nan','cursor_shape'):
            changed=banks.copy();initial=cursors.copy()
            if bad=='shape':changed=changed[:,:,:-1]
            if bad=='range':changed[0,0,0]=255
            if bad=='count':changed=np.repeat(banks[:1],65,axis=0)
            if bad=='nan':initial[0,0]=np.nan
            if bad=='cursor_shape':initial=initial[:,:-1]
            with pytest.raises(ValueError):session.configure_permutations(changed,initial)
        elif bad=='after':
            session.step()
            with pytest.raises(ValueError):session.configure_permutations(banks,cursors)
        elif bad=='mvs':
            pointers=(ct.POINTER(ct.c_uint8)*4)(*[b.ctypes.data_as(ct.POINTER(ct.c_uint8)) for b in banks])
            lambdas=np.ones(4,np.float32);valid=np.ones(4,np.uint8);error=ct.create_string_buffer(2048)
            assert session._lib.cbm_greedy_session_set_permutations(session._handle,4,pointers,None,
                lambdas.ctypes.data_as(ct.POINTER(ct.c_float)),valid.ctypes.data_as(ct.POINTER(ct.c_uint8)),error,len(error))
            assert b'MVS' in error.value
        else:
            session.configure_permutations(banks,cursors)
            if bad=='repeat':
                with pytest.raises(ValueError):session.configure_permutations(banks,cursors)
            elif bad=='index':
                with pytest.raises(ValueError):session.select_permutation(4)
            else:
                output=np.empty_like(cursors);error=ct.create_string_buffer(2048)
                assert session._lib.cbm_greedy_session_copy_permutation_state(session._handle,3,
                    output.ctypes.data_as(ct.POINTER(ct.c_float)),None,None,error,len(error))
                assert b'too small' in error.value


@pytest.mark.parametrize('policy', POLICIES)
@pytest.mark.parametrize('empty', [False,True])
def test_depth_zero_or_no_candidates_update_every_cursor(policy,empty):
    args,banks,cursors=inputs(policy)
    if empty:
        args.update(candidate_features=np.empty(0,np.uint32),candidate_bins=np.empty(0,np.uint32),candidate_types=np.empty(0,np.uint8))
    else:args['depth']=0
    compare_walk(args,banks,cursors,'all')


@pytest.mark.parametrize('policy', ['Lossguide','Region'])
@pytest.mark.parametrize('selected', [0,3])
def test_fixed_tree_replay_preserves_paths_beyond_symmetric_depth(policy,selected):
    from catboost_metal._greedy_inference import tree_depth
    positive=40;rows=2*positive
    bins=np.zeros((positive,rows),np.uint8);bins[np.arange(positive),np.arange(positive)]=1
    targets=np.r_[np.ones(positive),-np.ones(positive)].astype(np.float32)
    args=dict(bins=bins,targets=targets,candidate_features=np.arange(positive,dtype=np.uint32),
        candidate_bins=np.zeros(positive,np.uint32),grow_policy=policy,depth=40,max_leaves=41,
        iterations=1,learning_rate=.25,l2_leaf_reg=0,score_function='L2')
    with _greedy.Session(**args) as one,_greedy.Session(**args) as many:
        many.configure_permutations(np.repeat(bins[None],4,axis=0));many.select_permutation(selected)
        a,b=one.step(),many.step();assert tree_depth(a)>16
        for key in ('nodes','leaf_values','leaf_weights'):np.testing.assert_array_equal(getattr(a,key),getattr(b,key))
        np.testing.assert_array_equal(many.permutation_state['predictions'],np.repeat(one.predictions()[None],4,axis=0))
