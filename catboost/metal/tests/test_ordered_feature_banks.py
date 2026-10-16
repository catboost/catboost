"""Permutation-dependent Ordered routing, independent solves and full recovery."""
import ctypes as ct
import numpy as np
import pytest
from catboost_metal import _ordered
from cuda_ordered_reference import train_reference
from test_ordered_training import apple_silicon, prohibit_cpu_training, compare
from test_ordered_groups import problem, SIZES
from test_ordered_onehot_runtime import CASES


def bank_problem(count=4, grouped=False, objective='RMSE', parameter=None, method='Newton', **extra):
    bins,y,cf,cb,config=problem(objective,parameter,method,categorical=True,permutation_count=count)
    config.update(extra)
    if not grouped:config.pop('group_sizes')
    rng=np.random.default_rng(493)
    banks=np.repeat(bins[None],count,axis=0)
    for bank in range(1,count):
        # Bank-dependent threshold and equality features, plus a shared column.
        banks[bank,0]=rng.integers(0,8,len(y),dtype=np.uint8)
        banks[bank,1]=rng.integers(0,8,len(y),dtype=np.uint8)
    config['permutation_bins']=banks
    return bins,y,cf,cb,config


@pytest.mark.parametrize('objective,parameter,method',CASES)
@pytest.mark.parametrize('score',['Cosine','NewtonCosine'])
@pytest.mark.parametrize('grouped',[False,True])
def test_every_prefix_uses_its_feature_history(objective,parameter,method,score,grouped):
    bins,y,cf,cb,config=bank_problem(7,grouped,objective,parameter,method,score_function=score)
    expected=train_reference(bins,y,cf,cb,**config)
    with _ordered.Session(bins,y,cf,cb,**config) as session:
        for _ in range(config['iterations']):session.step()
        actual=session.result();state=session.state()
    compare(actual,expected)
    np.testing.assert_array_equal(actual.split_types,expected['split_types'])
    np.testing.assert_allclose(state['cursors'],expected['cursors'],rtol=2e-5,atol=5e-6)
    # Estimation bank routing must differ from merely applying bank-zero bins.
    assert not np.array_equal(config['permutation_bins'][0],config['permutation_bins'][-1])


@pytest.mark.parametrize('count',[4,7,64])
@pytest.mark.parametrize('sampler',['No','Bayesian','Bernoulli','Poisson','MVS'])
@pytest.mark.parametrize('backtracking',['No','AnyImprovement','Armijo'])
@pytest.mark.parametrize('grouped',[False,True])
def test_banked_snapshots_preserve_every_cursor_and_sampler(count,sampler,backtracking,grouped):
    bins,y,cf,cb,config=bank_problem(count,grouped,'Logloss',bootstrap_type=sampler,subsample=.7,
        random_strength=.8,leaf_estimation_backtracking=backtracking,leaf_estimation_iterations=3)
    with _ordered.Session(bins,y,cf,cb,**config) as session:
        session.step();saved=session.state();steps=[session.step() for _ in range(2)]
        final=session.state();pred=session.predictions()
    with _ordered.Session(bins,y,cf,cb,**(config|dict(iterations=2,initial_state=saved))) as restored:
        for step in steps:
            actual=restored.step()
            for name in ('depth','split_features','split_bins','split_types','leaf_values','leaf_weights','loss'):
                np.testing.assert_array_equal(getattr(actual,name),getattr(step,name))
        np.testing.assert_array_equal(restored.predictions(),pred)
        np.testing.assert_array_equal(restored.state()['cursors'],final['cursors'])
        assert restored.state()['mvs_lambda']==final['mvs_lambda']


@pytest.mark.parametrize('objective',['RMSE','Logloss','Quantile','MAPE'])
@pytest.mark.parametrize('backtracking',['No','AnyImprovement','Armijo'])
def test_banked_backtracking_and_exact_leaves_match_equations(objective,backtracking):
    exact=objective in ('Quantile','MAPE')
    bins,y,cf,cb,config=bank_problem(4,True,objective,.6 if objective=='Quantile' else None,
        'Exact' if exact else 'Newton',leaf_estimation_backtracking=backtracking,leaf_estimation_iterations=5)
    expected=train_reference(bins,y,cf,cb,**config)
    with _ordered.Session(bins,y,cf,cb,**config) as session:
        for _ in range(config['iterations']):session.step()
        compare(session.result(),expected)
        np.testing.assert_allclose(session.state()['cursors'],expected['cursors'],rtol=2e-5,atol=5e-6)


@pytest.mark.parametrize('count',[1,4,64])
@pytest.mark.parametrize('grouped',[False,True])
def test_identical_banks_keep_legacy_state_identity(count,grouped):
    bins,y,cf,cb,config=bank_problem(count,grouped)
    config.pop('permutation_bins')
    with _ordered.Session(bins,y,cf,cb,**config) as old:
        old.step();saved=old.state();step=old.step()
    with _ordered.Session(bins,y,cf,cb,**(config|dict(iterations=1,initial_state=saved,
        permutation_bins=np.repeat(bins[None],count,axis=0)))) as new:
        actual=new.step()
        np.testing.assert_array_equal(actual.leaf_values,step.leaf_values)
        assert new.state()['fingerprint']==saved['fingerprint']


@pytest.mark.parametrize('change',['shape','float','negative','range','first','identity'])
def test_invalid_or_changed_banks_fail_before_gpu(monkeypatch,change):
    bins,y,cf,cb,config=bank_problem()
    if change=='identity':
        with _ordered.Session(bins,y,cf,cb,**config) as session:
            session.step();config['initial_state']=session.state()
        config['permutation_bins'][-1,0,0]^=1
    elif change=='shape':config['permutation_bins']=config['permutation_bins'][1:]
    elif change=='float':config['permutation_bins']=config['permutation_bins'].astype(float)
    elif change in ('negative','range'):
        config['permutation_bins']=config['permutation_bins'].astype(np.int32)
        config['permutation_bins'][-1,0,0]=-1 if change=='negative' else 256
    else:config['permutation_bins'][0,0,0]^=1
    monkeypatch.setattr(_ordered,'build_library',lambda:pytest.fail('invalid input reached GPU'))
    with pytest.raises(ValueError):_ordered.Session(bins,y,cf,cb,**config)


@pytest.mark.parametrize('banks,cells',[(0,0),(2,12),(65,390),(4,1),(4,24)])
def test_counted_c_abi_rejects_invalid_geometry(banks,cells):
    # All input pointers are null: validation must precede every input read.
    p=_ordered.Params();p.rows=6;p.features=1;p.permutations=4
    lib=_ordered._load(_ordered.build_library());error=ct.create_string_buffer(4096);handle=ct.c_void_p(123)
    result=lib.cbm_ordered_session_create_banked(ct.byref(p),banks,cells,None,None,None,None,None,None,None,None,
        0,None,2.,ct.byref(handle),error,len(error))
    assert result and not handle.value and error.value


@pytest.mark.parametrize('strength',[0.,.5,2.,7.])
@pytest.mark.parametrize('grouped',[False,True])
@pytest.mark.parametrize('feature_weights',[[1.,1.,1.],[.3,2.,.6]])
def test_simple_ctr_penalties_remain_after_selection(strength,grouped,feature_weights):
    bins,y,cf,cb,config=bank_problem(4,grouped,ctr_unique_values=[19,3,0],model_size_reg=strength,
        feature_weights=feature_weights,iterations=7)
    expected=train_reference(bins,y,cf,cb,**config)
    with _ordered.Session(bins,y,cf,cb,**config) as session:
        session.step();saved=session.state()
        for _ in range(6):session.step()
        compare(session.result(),expected);final=session.state()
    with _ordered.Session(bins,y,cf,cb,**(config|dict(iterations=6,initial_state=saved))) as restored:
        for _ in range(6):restored.step()
        np.testing.assert_array_equal(restored.state()['cursors'],final['cursors'])
    # Counts are immutable simple-CTR metadata; FeatureParallel marks only
    # dynamic tree CTRs used, unlike DocParallel's per-split exemption.
    assert 'used_ctrs' not in final


@pytest.mark.parametrize('key,value',[('ctr_unique_values',[1,2]),('ctr_unique_values',[-1,2,3]),
    ('model_size_reg',float('nan')),('model_size_reg',-.5),('feature_weights',[1,-1,1])])
def test_invalid_penalties_rejected_before_gpu(monkeypatch,key,value):
    bins,y,cf,cb,config=bank_problem(ctr_unique_values=[4,8,0]);config[key]=value
    monkeypatch.setattr(_ordered,'build_library',lambda:pytest.fail('invalid penalty reached GPU'))
    with pytest.raises(ValueError):_ordered.Session(bins,y,cf,cb,**config)
