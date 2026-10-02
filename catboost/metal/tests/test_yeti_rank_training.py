"""Resident YetiRank tree search and leaf walk against explicit CUDA equations."""
import platform
import numpy as np
import pytest
from catboost_metal import _native, _yeti
from cuda_reference import _score_children
from test_yeti_rank_kernels import reference
from test_pairwise_training import leaf_ids

pytestmark=pytest.mark.skipif(platform.system()!='Darwin' or platform.machine()!='arm64',reason='requires Apple Silicon GPU')

@pytest.fixture(autouse=True)
def no_cpu_fit(monkeypatch):
    from catboost import CatBoost
    def forbidden(*args,**kwargs):raise AssertionError('No CPU CatBoost training')
    monkeypatch.setattr(CatBoost,'_fit',forbidden)


def problem(**extra):
    rng=np.random.default_rng(78321);rows=67
    bins=rng.integers(0,4,(3,rows),dtype=np.uint8)
    y=np.clip(.1+.6*(bins[0]>1)+rng.uniform(0,.3,rows),0,1).astype(np.float32)
    weights=rng.uniform(.2,2,rows).astype(np.float32);weights[::17]=0
    result=dict(bins=bins,targets=y,candidate_features=np.repeat(np.arange(3,dtype=np.uint32),3),
        candidate_bins=np.tile(np.arange(3,dtype=np.uint32),3),group_offsets=np.array([0,19,40,rows],np.uint32),
        sample_weight=weights,initial_predictions=rng.normal(0,.4,rows).astype(np.float32),
        iterations=3,depth=2,score_function='Cosine',learning_rate=.17,l2_leaf_reg=.2,
        permutations=7,decay=.85,leaf_estimation_iterations=3)
    result.update(extra);return result


def terms(args,point,seed):
    return reference(args['targets'],args['sample_weight'],point,args['group_offsets'],
        permutations=args['permutations'],decay=args['decay'],seed=seed,
        center_rows=len(args['group_offsets'])-1 if args.get('legacy_prefix_centering') else None)[1]


def oracle_leaves(args,cursor,ids,count,seeds):
    value=np.zeros(count,np.float32)
    weights=np.bincount(ids,weights=args['sample_weight'].astype(float),minlength=count)
    for seed in seeds:
        grad,mass=terms(args,np.float32(cursor+value[ids]),seed).T
        g=np.bincount(ids,weights=grad.astype(float),minlength=count)
        h=np.bincount(ids,weights=mass.astype(float),minlength=count)
        value=np.float32(value+g/(h+args['l2_leaf_reg']+1e-20))
        value[weights<1e-20]=0
    value=np.float32(value-np.float32(value.mean(dtype=float)))
    return np.float32(value*np.float32(args['learning_rate'])),weights


@pytest.mark.parametrize('score',['L2','Cosine','NewtonL2','NewtonCosine','SolarL2','LOOL2','SatL2'])
@pytest.mark.parametrize('leaf_iterations',[1,3])
@pytest.mark.parametrize('legacy',[False,True])
def test_each_tree_uses_its_explicit_leaf_oracle_draws(score,leaf_iterations,legacy):
    args=problem(score_function=score,leaf_estimation_iterations=leaf_iterations,legacy_prefix_centering=legacy)
    cursor=args['initial_predictions'].copy()
    with _yeti.Session(**args) as session:
        for iteration in range(args['iterations']):
            seeds=[(0xdef01234<<32)+47+iteration*11+i for i in range(leaf_iterations+1+int(leaf_iterations>1))]
            tree=session.step(seeds);ids=leaf_ids(tree,args['bins'])
            expected,weights=oracle_leaves(args,cursor,ids,1<<tree.depth,seeds[1:1+leaf_iterations])
            np.testing.assert_allclose(tree.leaf_values,expected,rtol=8e-5,atol=8e-7)
            np.testing.assert_allclose(tree.leaf_weights,weights,rtol=3e-6,atol=1e-5)
            assert abs(tree.leaf_values.mean(dtype=float))<1e-7
            cursor=np.float32(cursor+tree.leaf_values[ids])
            np.testing.assert_array_equal(session.predictions(),cursor)
            assert tree.loss==0.  # CUDA's stochastic oracle reports no scalar objective.


@pytest.mark.parametrize('score',['L2','Cosine','NewtonL2','NewtonCosine'])
def test_weak_target_uses_incident_mass_for_both_score_families(score):
    args=problem(score_function=score,iterations=1)
    g,h=terms(args,args['initial_predictions'],17).T
    ids=np.zeros(len(g),np.uint32);chosen=[]
    for level in range(args['depth']):
        scores=[]
        for f,b in zip(args['candidate_features'],args['candidate_bins']):
            trial=ids*2+(args['bins'][f]>b)
            sums=np.bincount(trial,weights=g.astype(float),minlength=2<<level)
            masses=np.bincount(trial,weights=h.astype(float),minlength=2<<level)
            scores.append(_score_children(sums,masses,args['l2_leaf_reg'],score.removeprefix('Newton')))
        winner=int(np.argmin(scores));split=(int(args['candidate_features'][winner]),int(args['candidate_bins'][winner]))
        if split in chosen:break
        chosen.append(split);ids|=(args['bins'][split[0]]>split[1]).astype(np.uint32)<<level
    with _yeti.Session(**args) as session:
        tree=session.step([17,18,19,20,21])
    assert list(zip(tree.split_features,tree.split_bins))==chosen


@pytest.mark.parametrize('kind',['No','Bayesian','Bernoulli','Poisson','MVS'])
@pytest.mark.parametrize('noise',[0.,.7])
def test_sampling_noise_and_restored_cursor_preserve_tree_sequences(kind,noise):
    args=problem(iterations=5,bootstrap_type=kind,random_seed=817,random_strength=noise)
    if kind in ('Bernoulli','Poisson','MVS'):args['subsample']=.7
    seeds=[[147+i*10+j for j in range(5)] for i in range(5)]
    with _yeti.Session(**args) as full:
        expected=[full.step(seed) for seed in seeds]
        final=full.predictions()
    with _yeti.Session(**{**args,'iterations':2}) as partial:
        for seed in seeds[:2]:partial.step(seed)
        cursor=partial.predictions();state=partial.bootstrap_state
    with _yeti.Session(**{**args,'iterations':3,'iteration_offset':2,'initial_predictions':cursor,
                           'initial_mvs_lambda':state['mvs_lambda']}) as resumed:
        for index,seed in enumerate(seeds[2:],2):
            tree=resumed.step(seed)
            np.testing.assert_array_equal(tree.split_features,expected[index].split_features)
            np.testing.assert_array_equal(tree.split_bins,expected[index].split_bins)
            np.testing.assert_array_equal(tree.leaf_values,expected[index].leaf_values)
        np.testing.assert_array_equal(resumed.predictions(),final)


def test_incremental_tree_search_uses_one_seed_schedule():
    args=problem(iterations=1,random_strength=.7)
    with _yeti.Session(**args) as direct:expected=direct.step([7,8,9,10,11])
    with _yeti.Session(**args) as staged:
        staged.begin_tree([7,8,9,10,11])
        while not staged.grow_tree()['finished']:pass
        tree=staged.finish_tree()
    np.testing.assert_array_equal(tree.leaf_values,expected.leaf_values)
    np.testing.assert_array_equal(tree.split_features,expected.split_features)


@pytest.mark.parametrize('option',[dict(leaf_estimation_method='Gradient'),dict(leaf_estimation_backtracking='Armijo'),
    dict(permutations=0),dict(decay=np.inf),dict(legacy_prefix_centering=2)])
def test_unsupported_objective_settings_rejected(option):
    with pytest.raises((ValueError,RuntimeError)):
        with _yeti.Session(**problem(**option)):pass


@pytest.mark.parametrize('seeds',[[1,2],[1,2,3,4,2**64],[1,2,3,4,-1],[True,2,3,4,5]])
def test_invalid_seed_schedule_fails_before_tree_start_and_can_be_corrected(seeds):
    with _yeti.Session(**problem(iterations=1)) as session:
        with pytest.raises(ValueError,match='uint64'):session.step(seeds)
        assert session.step([1,2,3,4,5]).depth>0


@pytest.mark.parametrize('depth', [0, 2, 16])
@pytest.mark.parametrize('empty', [False, True])
def test_delayed_leaf_seeds_match_full_schedule_at_depth_and_candidate_boundaries(depth, empty):
    args = problem(iterations=1, depth=depth)
    if empty:
        args.update(candidate_features=np.empty(0, np.uint32), candidate_bins=np.empty(0, np.uint32))
    with _yeti.Session(**args) as direct:
        expected = direct.step([7, 8, 9, 10, 11])
        prediction = direct.predictions()
    with _yeti.Session(**args) as staged:
        staged.begin_tree([7])
        while not staged.grow_tree()['finished']: pass
        tree = staged.finish_tree([8, 9, 10, 11])
        np.testing.assert_array_equal(staged.predictions(), prediction)
    np.testing.assert_array_equal(tree.leaf_values, expected.leaf_values)
    np.testing.assert_array_equal(tree.split_features, expected.split_features)
