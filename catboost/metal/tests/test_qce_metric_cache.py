"""Tracker metric workspaces reuse metadata across iterations and alpha values."""
import os
import numpy as np
import pytest
from catboost import CatBoost,CatBoostRanker,Pool
from catboost_metal import CatBoostMetalRanker
from catboost_metal import _query_cross_entropy as qce
from test_query_cross_entropy_lifecycle import inputs,params
from test_native_pair_api import StopAfter

@pytest.fixture(autouse=True)
def forbid_cpu_fit(monkeypatch):
    original=CatBoost._fit
    def fit(model,*args,**kwargs):
        assert model.get_params().get('task_type')=='GPU','CPU fitting is forbidden'
        return original(model,*args,**kwargs)
    monkeypatch.setattr(CatBoost,'_fit',fit)


@pytest.fixture
def tracked(monkeypatch):
    sessions=[];implementation=qce.MetricSession
    class Observed(implementation):
        def __init__(self,*args,**kwargs):
            super().__init__(*args,**kwargs);sessions.append(self);self.closed_count=0
        def close(self):
            if self._handle.value:self.closed_count+=1
            super().close()
    monkeypatch.setattr(qce,'MetricSession',Observed);return sessions


@pytest.mark.parametrize('validation',[False,True])
@pytest.mark.parametrize('selected',[None,'QueryCrossEntropy:alpha=.2;use_weights=false','PFound'])
def test_standalone_metric_workspace_is_created_once_and_closed(tracked,validation,selected):
    args,x,y,groups,weights=inputs();options=params(iterations=5,eval_metric=selected)
    fit=dict(group_id=groups,sample_weight=weights,use_best_model=False)
    if validation:fit['eval_set']=(x,y,groups,weights)
    model=CatBoostMetalRanker(**options).fit(x,y,**fit)
    required=validation or selected is not None and selected.startswith('QueryCrossEntropy')
    assert len(tracked)==int(required)
    if required:
        expected=5*(2 if validation and selected is not None and selected.startswith('QueryCrossEntropy') else 1)
        assert tracked[0].evaluations==expected and not tracked[0]._handle.value and tracked[0].closed_count==1
        scales=qce.select_scales('0,0:.8 4,2:1.6',y,args['group_offsets'])
        dataset='validation' if validation else 'learn'
        for name,values in model.evals_result_[dataset].items():
            if name.startswith('QueryCrossEntropy'):
                alpha=qce.parse_description(name,metric=True).get('alpha',.95)
                expected=qce.metric(model.predict(x),y,weights,args['group_offsets'],alpha=alpha,query_scales=scales)
                assert values[-1]==pytest.approx(expected,abs=3e-7)


def test_workspace_closes_when_user_callback_raises(tracked):
    _,x,y,groups,weights=inputs()
    def fail(event):raise RuntimeError('callback sentinel')
    with pytest.raises(RuntimeError,match='callback sentinel'):
        CatBoostMetalRanker(**params()).fit(x,y,group_id=groups,sample_weight=weights,eval_set=(x,y,groups,weights),callback=fail)
    assert len(tracked)==1 and tracked[0].closed_count==1 and not tracked[0]._handle.value


def test_resumed_tracker_uses_a_fresh_workspace_with_identical_history(tracked,tmp_path):
    _,x,y,groups,weights=inputs();opts=params(iterations=5,eval_metric='QueryCrossEntropy:alpha=.2')
    fit=dict(group_id=groups,sample_weight=weights,eval_set=(x,y,groups,weights),use_best_model=False)
    direct=CatBoostMetalRanker(**opts).fit(x,y,**fit);snap=tmp_path/'metric.snapshot'
    CatBoostMetalRanker(**opts).fit(x,y,**fit,save_snapshot=True,snapshot_file=snap,snapshot_interval=0,callback=lambda event:event.iteration<2)
    resumed=CatBoostMetalRanker(**opts).fit(x,y,**fit,save_snapshot=True,snapshot_file=snap,snapshot_interval=0)
    assert len(tracked)==3 and [s.evaluations for s in tracked]==[10,4,6]
    assert all(s.closed_count==1 and not s._handle.value for s in tracked)
    assert direct.evals_result_==resumed.evals_result_
    np.testing.assert_array_equal(direct.predict(x),resumed.predict(x))


@pytest.mark.skipif(os.environ.get('CATBOOST_NATIVE_METAL_QUERY_TESTS')!='1',reason='rebuilt native tracker required')
@pytest.mark.parametrize('kind',['No','Bernoulli'])
def test_native_multiple_datasets_and_metric_alphas_keep_original_target_semantics(tmp_path,kind):
    _,x,y,groups,weights=inputs();loss='QueryCrossEntropy:alpha=.7;raw_values_scale=0,0:.8 4,2:1.6'
    datasets=[(x,y,weights),(x+.125,np.float32(1-y),np.float32(weights*.3))]
    pools=[Pool(a,b,weight=w,group_id=groups) for a,b,w in datasets]
    options=dict(task_type='GPU',loss_function=loss,eval_metric='QueryCrossEntropy:alpha=.2',
        custom_metric=['QueryCrossEntropy:alpha=0','QueryCrossEntropy:alpha=1;use_weights=false'],
        iterations=5,depth=2,border_count=8,bootstrap_type=kind,random_strength=0,random_seed=738,
        verbose=False,allow_writing_files=False,metric_period=1)
    if kind=='Bernoulli':options['subsample']=.7
    direct=CatBoostRanker(**options).fit(pools[0],eval_set=pools,use_best_model=False)
    saved=options|dict(save_snapshot=True,snapshot_interval=0,snapshot_file='cache.snapshot',allow_writing_files=True,train_dir=str(tmp_path))
    partial=CatBoostRanker(**saved).fit(pools[0],eval_set=pools,use_best_model=False,callbacks=[StopAfter()]);assert partial.tree_count_==3
    resumed=CatBoostRanker(**saved).fit(pools[0],eval_set=pools,use_best_model=False)
    assert resumed.evals_result_==direct.evals_result_
    for index,(a,b,w) in enumerate(datasets):
        offsets=np.r_[0,np.flatnonzero(groups[1:]!=groups[:-1])+1,len(groups)].astype(np.uint32)
        scales=qce.select_scales('0,0:.8 4,2:1.6',b,offsets)
        for name,values in direct.evals_result_[f'validation_{index}'].items():
            alpha=qce.parse_description(name,metric=True).get('alpha',.95)
            expected=qce.metric(direct.predict(a,task_type='GPU'),b,w,offsets,alpha=alpha,query_scales=scales)
            assert values[-1]==pytest.approx(expected,abs=3e-7)
        np.testing.assert_array_equal(direct.predict(a,task_type='GPU'),resumed.predict(a,task_type='GPU'))
