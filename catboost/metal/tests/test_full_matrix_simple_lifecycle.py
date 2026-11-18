"""Standalone Simple leaves preserve matrix weights through replay and export."""
import numpy as np
import pytest
from catboost import CatBoost,CatBoostRanker
from catboost_metal import CatBoostMetalRanker
from test_query_cross_entropy_lifecycle import inputs as qce_inputs,params as qce_params
from test_pairwise_matrix_lifecycle import inputs as pair_inputs,params as pair_params

@pytest.fixture(autouse=True)
def no_cpu_fit(monkeypatch):
    def forbidden(*args,**kw):raise AssertionError('No CPU CatBoost fitting')
    monkeypatch.setattr(CatBoost,'_fit',forbidden)


@pytest.mark.parametrize('target',['pair','qce'])
@pytest.mark.parametrize('kind',['No','Bernoulli'])
@pytest.mark.parametrize('mode',['No','AnyImprovement','Armijo'])
def test_simple_public_snapshot_export_and_original_metric_weights(tmp_path,target,kind,mode):
    if target=='pair':
        bins,y,_,_,_,groups,pairs,edge_weights,weights=pair_inputs();x=bins.T
        options=pair_params();fit=dict(pairs=pairs,pairs_weight=edge_weights,eval_pairs=pairs,eval_pairs_weight=edge_weights)
    else:
        args,x,y,groups,weights=qce_inputs();options=qce_params();fit={}
    options.update(leaf_estimation_method='Simple',leaf_estimation_iterations=1,leaf_estimation_backtracking=mode,bootstrap_type=kind)
    if kind!='No':options['subsample']=.7
    fit.update(group_id=groups,sample_weight=weights,eval_set=(x,y,groups,weights),use_best_model=False)
    direct=CatBoostMetalRanker(**options).fit(x,y,**fit)
    snapshot=tmp_path/'simple.snapshot'
    partial=CatBoostMetalRanker(**options).fit(x,y,**fit,save_snapshot=True,snapshot_file=snapshot,snapshot_interval=0,
        callback=lambda event:event.iteration<2)
    assert partial.tree_count_==2
    resumed=CatBoostMetalRanker(**options).fit(x,y,**fit,save_snapshot=True,snapshot_file=snapshot,snapshot_interval=0)
    for attr in ('get_leaf_values','get_leaf_weights'):
        np.testing.assert_array_equal(getattr(resumed._model,attr)(),getattr(direct._model,attr)())
    assert resumed.evals_result_==direct.evals_result_
    np.testing.assert_array_equal(resumed.predict(x),direct.predict(x))
    assert not np.isclose(direct._model.get_leaf_weights().sum(),weights.sum()*direct.tree_count_)
    for format in ('cbm','json'):
        path=tmp_path/('simple.'+format);resumed.save_model(path,format=format)
        model=CatBoostRanker().load_model(str(path),format=format)
        np.testing.assert_allclose(model.predict(x),direct.predict(x),rtol=1e-10,atol=1e-10)
        np.testing.assert_allclose(model.predict(x,task_type='GPU'),direct.predict(x),rtol=1e-10,atol=1e-10)
        if format=='cbm':np.testing.assert_array_equal(model.get_leaf_weights(),direct._model.get_leaf_weights())
        else:np.testing.assert_array_equal(model.get_leaf_weights().astype(np.float32),direct._model.get_leaf_weights().astype(np.float32))
