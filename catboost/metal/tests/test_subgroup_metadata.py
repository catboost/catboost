"""Stored subgroup identity, PFound semantics and actual Metal lifecycle replay."""
import os
import numpy as np
import pytest
from catboost import CatBoost,CatBoostError,Pool,_catboost
from catboost.utils import eval_metric
from catboost_metal import CatBoostMetalRanker
from catboost_metal._training import _shared_metric

pytestmark=pytest.mark.skipif(os.environ.get('CATBOOST_NATIVE_METAL_TESTS')!='1',reason='requires rebuilt subgroup metadata accessors')

@pytest.fixture(autouse=True)
def no_cpu_fit(monkeypatch):
    monkeypatch.setattr(CatBoost,'_fit',lambda *args,**kwargs:pytest.fail('CPU CatBoost fitting is forbidden'))


def problem():
    rng=np.random.default_rng(93475);groups=np.repeat(np.arange(8),6);x=rng.normal(size=(48,3)).astype(np.float32)
    y=np.clip(.4+.15*x[:,0]-.12*x[:,1],.01,.99).astype(np.float32);w=rng.uniform(.2,2,48).astype(np.float32)
    sub=np.tile(np.array(['a','a','b','c','b','c']),8);gw=np.repeat(np.linspace(.5,2,8,dtype=np.float32),6)
    return x,y,groups,w,sub,gw


@pytest.mark.parametrize('tokens',[
    ['α','α','b','c','b','c'],[0,0,17,2**63-1,17,2**63-1],[-3,-3,0,17,0,17],['0',0,'17',17,'x','x'],
])
@pytest.mark.parametrize('quantized',[False,True])
def test_pool_hashes_preserve_tokens_slices_quantization_and_copy(tokens,quantized):
    x=np.arange(12,dtype=np.float32).reshape(6,2);pool=Pool(x,group_id=[0,0,0,1,1,1],subgroup_id=tokens)
    expected=_catboost._hash_subgroup_ids(tokens)
    if quantized:pool.quantize(border_count=4)
    assert pool.get_subgroup_id_hash().dtype==np.uint32
    np.testing.assert_array_equal(pool.get_subgroup_id_hash(),expected)
    np.testing.assert_array_equal(pool.slice([3,4,5,0,1,2]).get_subgroup_id_hash(),expected[[3,4,5,0,1,2]])
    np.testing.assert_array_equal(pool.slice([3,4,5]).get_subgroup_id_hash(),expected[3:])
    copy=pool.get_subgroup_id_hash();copy[:]=0;np.testing.assert_array_equal(pool.get_subgroup_id_hash(),expected)


def test_pool_without_subgroups_and_updated_subgroups():
    pool=Pool(np.arange(12,dtype=np.float32).reshape(6,2),group_id=[0]*3+[1]*3)
    assert pool.get_subgroup_id_hash() is None
    pool.set_subgroup_id(['a','b','a','c','c','d'])
    np.testing.assert_array_equal(pool.get_subgroup_id_hash(),_catboost._hash_subgroup_ids(['a','b','a','c','c','d']))


def independent_pfound(raw,y,groups,weights,subgroups,top,decay,weighted=True):
    values=[];mass=[]
    for group in np.unique(groups):
        rows=np.flatnonzero(groups==group);ordered=sorted(rows,key=lambda i:(-raw[i],y[i]))[:top]
        found=0.;look=1.;seen=set()
        for row in ordered:
            if subgroups[row] in seen:continue
            seen.add(subgroups[row]);found+=look*float(y[row]);look*=(1-float(y[row]))*decay
        values.append(found);mass.append(float(weights[rows[0]]) if weighted else 1.)
    return np.average(values,weights=mass)


@pytest.mark.parametrize('top,decay',[(1,.85),(3,.7),(6,1)])
@pytest.mark.parametrize('weighted',[False,True])
def test_shared_pfound_skips_subgroups_inside_original_top_window(top,decay,weighted):
    x,y,g,w,sub,gw=problem();raw=x[:,0].astype(float);raw[:3]=.5;weights=np.float32(w*gw)
    offsets=np.r_[0,np.arange(6,49,6)].astype(np.uint32);hashes=_catboost._hash_subgroup_ids(sub)
    name=f'PFound:top={top};decay={decay};use_weights={str(weighted).lower()}'
    actual=_shared_metric(name,raw,y,weights,offsets,subgroup_hashes=hashes)
    assert actual==pytest.approx(independent_pfound(raw,y,g,weights,sub,top,decay,weighted),abs=2e-15)


def test_prehashed_subgroups_are_not_rehashed():
    # Distinct uint32 IDs can collide if their decimal strings are hashed again.
    a,b=47220,111524
    np.testing.assert_array_equal(_catboost._hash_subgroup_ids([a,b]),[1651280859,1651280859])
    args=([.5,.5],[1.,0.],'PFound');ordinary=eval_metric(*args,group_id=[0,0],subgroup_id=[int(a),int(b)])[0]
    actual=_shared_metric('PFound',args[1],args[0],None,[0,2],subgroup_hashes=np.array([a,b],np.uint32))
    assert ordinary==.5 and actual==pytest.approx(.5+.5*.85*.5,abs=1e-15)


@pytest.mark.parametrize('bad',[[0.5,1.5],[-1,0],[0,2**32],[False,True],[[1],[2]],[0]])
def test_private_metric_rejects_invalid_stored_hashes(bad):
    with pytest.raises(CatBoostError,match='(?i)(hash|subgroup|size)'):
        _catboost._eval_metric_util([[.2,.6]],[[.5,.1]],'PFound',None,[0,0],None,bad,None,1,True)


def options(loss,**extra):
    result=dict(loss_function=loss,iterations=5,depth=2,border_count=8,bootstrap_type='No',random_strength=0,
        random_seed=739,learning_rate=.13,eval_metric='PFound:top=4;decay=.7',leaf_estimation_backtracking='No')
    result.update(extra);return result


def fit_options(loss,x,y,g,w,sub,gw):
    paired=loss.startswith('PairLogit')
    result=dict(group_id=g,subgroup_id=sub,eval_set=(x,y,g,None if loss=='PairLogit' else w,None if loss=='PairLogit' else gw,sub),use_best_model=False)
    if loss!='PairLogit':result.update(sample_weight=w,group_weight=gw)
    if paired:
        pairs=np.array([(6*i+1,6*i) for i in range(8)])
        result.update(pairs=pairs,eval_pairs=pairs)
    return result


@pytest.mark.parametrize('loss',['QueryRMSE','QuerySoftMax','PairLogit','PairLogitPairwise','QueryCrossEntropy','YetiRank','YetiRankPairwise'])
def test_subgroups_drive_metric_history_best_model_and_exact_snapshot(tmp_path,loss):
    x,y,g,w,sub,gw=problem();fit=fit_options(loss,x,y,g,w,sub,gw);opts=options(loss)
    full=CatBoostMetalRanker(**opts).fit(x,y,**fit);snap=tmp_path/'subgroups.snapshot'
    CatBoostMetalRanker(**opts).fit(x,y,**fit,save_snapshot=True,snapshot_file=snap,snapshot_interval=0,callback=lambda event:event.iteration<2)
    resumed=CatBoostMetalRanker(**opts).fit(x,y,**fit,save_snapshot=True,snapshot_file=snap)
    np.testing.assert_array_equal(full.predict(x),resumed.predict(x));assert full.evals_result_==resumed.evals_result_
    with np.load(snap,allow_pickle=False) as archive:assert all(archive[n].dtype.kind!='O' for n in archive.files)
    weights=np.ones(len(x),np.float32) if loss=='PairLogit' else np.float32(w*gw)
    for i,value in enumerate(full.evals_result_['validation'][opts['eval_metric']],1):
        raw=full.predict(x,ntree_end=i)
        assert value==pytest.approx(independent_pfound(raw,y,g,weights,sub,4,.7),abs=2e-7)
    if loss.startswith('YetiRank'):
        for i,value in enumerate(full.evals_result_['learn']['PFound'],1):
            assert value==pytest.approx(independent_pfound(full.predict(x,ntree_end=i),y,g,weights,sub,6,.85),abs=2e-7)
        np.testing.assert_allclose(full.loss_history_[1:],full.evals_result_['learn']['PFound'],rtol=1e-7)
    best=CatBoostMetalRanker(**opts).fit(x,y,**(fit|{'use_best_model':True}))
    selected=int(np.argmax(full.evals_result_['validation'][opts['eval_metric']]))
    assert best.tree_count_==selected+1 and best.best_iteration_==selected
    np.testing.assert_array_equal(best.predict(x),full.predict(x,ntree_end=selected+1))


@pytest.mark.parametrize('loss',['YetiRank','YetiRankPairwise','QueryRMSE'])
@pytest.mark.parametrize('dataset',['learn','validation'])
def test_snapshot_rejects_changed_subgroup_metadata(tmp_path,loss,dataset):
    x,y,g,w,sub,gw=problem();fit=fit_options(loss,x,y,g,w,sub,gw);snap=tmp_path/'metadata.snapshot';opts=options(loss)
    CatBoostMetalRanker(**opts).fit(x,y,**fit,save_snapshot=True,snapshot_file=snap,snapshot_interval=0,callback=lambda event:event.iteration<2)
    changed=sub.copy();changed[0]='z'
    if dataset=='learn':fit['subgroup_id']=changed
    else:fit['eval_set']=(x,y,g,w,gw,changed)
    with pytest.raises(ValueError,match='(?i)(snapshot|fingerprint)'):
        CatBoostMetalRanker(**opts).fit(x,y,**fit,save_snapshot=True,snapshot_file=snap)


@pytest.mark.parametrize('loss',['YetiRank','YetiRankPairwise','QueryRMSE'])
def test_pool_and_array_subgroups_preserve_training_and_evaluation(loss):
    x,y,g,w,sub,gw=problem();array=CatBoostMetalRanker(**options(loss)).fit(x,y,**fit_options(loss,x,y,g,w,sub,gw))
    pool=Pool(x,y,group_id=g,subgroup_id=sub,weight=np.float32(w*gw))
    actual=CatBoostMetalRanker(**options(loss)).fit(pool,eval_set=pool,use_best_model=False)
    np.testing.assert_array_equal(actual.predict(x),array.predict(x));assert actual.evals_result_==array.evals_result_
    # Metadata changes the tracker only; ranking target generation is unchanged.
    no_sub=CatBoostMetalRanker(**options(loss)).fit(x,y,group_id=g,sample_weight=w,group_weight=gw,use_best_model=False)
    np.testing.assert_array_equal(no_sub.predict(x),array.predict(x))


@pytest.mark.parametrize('loss',['QueryRMSE','QuerySoftMax','PairLogit','PairLogitPairwise','QueryCrossEntropy','YetiRank','YetiRankPairwise'])
def test_learn_only_selection_metric_keeps_subgroup_metadata(loss):
    x,y,g,w,sub,gw=problem();fit=fit_options(loss,x,y,g,w,sub,gw);fit.pop('eval_set');fit.pop('eval_pairs',None)
    model=CatBoostMetalRanker(**options(loss)).fit(x,y,**fit)
    weights=np.ones(len(x),np.float32) if loss=='PairLogit' else np.float32(w*gw)
    for i,value in enumerate(model.evals_result_['learn']['PFound:top=4;decay=.7'],1):
        expected=independent_pfound(model.predict(x,ntree_end=i),y,g,weights,sub,4,.7)
        assert value==pytest.approx(expected,abs=2e-7)


@pytest.mark.parametrize('bad',[['a'],[['a']]*48,[0.5]*48,[True]*48,[None]*48])
def test_array_subgroup_validation(bad):
    x,y,g,w,sub,gw=problem()
    with pytest.raises(ValueError,match='(?i)subgroup'):
        CatBoostMetalRanker(**options('YetiRank')).fit(x,y,group_id=g,subgroup_id=bad)


def test_eval_subgroup_keywords_tuple_conflicts_and_pool_ownership():
    x,y,g,w,sub,gw=problem();opts=options('YetiRankPairwise');fit=fit_options('YetiRankPairwise',x,y,g,w,sub,gw)
    expected=CatBoostMetalRanker(**opts).fit(x,y,**fit)
    actual=CatBoostMetalRanker(**opts).fit(x,y,**(fit|{'eval_set':(x,y,g,w,gw),'eval_subgroup_id':sub}))
    assert actual.evals_result_==expected.evals_result_
    with pytest.raises(ValueError,match='tuple or keywords'):
        CatBoostMetalRanker(**opts).fit(x,y,**(fit|{'eval_subgroup_id':sub}))
    with pytest.raises(ValueError,match='eval_set'):
        CatBoostMetalRanker(**opts).fit(x,y,group_id=g,eval_subgroup_id=sub)
    pool=Pool(x,y,group_id=g,subgroup_id=sub)
    with pytest.raises(ValueError,match='supplied Pool'):
        CatBoostMetalRanker(**opts).fit(pool,subgroup_id=sub)
    with pytest.raises(ValueError,match='supplied Pool'):
        CatBoostMetalRanker(**opts).fit(pool,eval_set=pool,eval_subgroup_id=sub)
