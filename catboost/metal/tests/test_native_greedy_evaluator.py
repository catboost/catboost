"""Native GPU application of variable trees; all training uses Metal."""
import json
import os

import numpy as np
import pytest
from catboost import CatBoost, Pool
from catboost_metal import CatBoostMetalRegressor, CatBoostMetalClassifier
from catboost_metal._greedy_model import dumps_model_json
from test_greedy_deep_model import document


pytestmark=pytest.mark.skipif(os.environ.get('CATBOOST_NATIVE_METAL_TESTS')!='1',
                            reason='requires the rebuilt native variable-tree GPU evaluator')


@pytest.fixture(autouse=True)
def no_cpu_training(monkeypatch):
    def forbidden(*args,**kwargs):raise AssertionError('No CPU CatBoost training is permitted')
    monkeypatch.setattr(CatBoost,'_fit',forbidden)


@pytest.mark.parametrize('policy',['Depthwise','Lossguide','Region'])
@pytest.mark.parametrize('classification',[False,True])
def test_native_gpu_predictions_for_grown_variable_forests(tmp_path,policy,classification):
    rng=np.random.default_rng(642);x=rng.normal(size=(517,4)).astype(np.float32)
    y=x[:,0]-.5*x[:,1]+(x[:,2]>.2)
    cls=CatBoostMetalClassifier if classification else CatBoostMetalRegressor
    model=cls(iterations=35,depth=5,grow_policy=policy,bootstrap_type='Bernoulli',subsample=.8,
              random_strength=.4,random_seed=63).fit(x,(y>0) if classification else y).to_catboost()
    for fmt in ('json','cbm'):
        path=tmp_path/('forest.'+fmt);model.save_model(path,format=fmt)
        restored=CatBoost().load_model(path,format=fmt)
        for kind in (['RawFormulaVal','Probability','LogProbability','Class'] if classification else ['RawFormulaVal']):
            for begin,end in [(0,0),(3,31)]:
                opts=dict(prediction_type=kind,ntree_start=begin,ntree_end=end)
                expected=restored.predict(x,**opts)
                np.testing.assert_allclose(restored.predict(x,task_type='GPU',**opts),expected,rtol=3e-12,atol=3e-12)
        border_path=tmp_path/'borders.tsv';restored.save_borders(border_path)
        pool=Pool(x);pool.quantize(input_borders=str(border_path))
        np.testing.assert_allclose(restored.predict(pool,task_type='GPU'),restored.predict(pool),rtol=3e-12,atol=3e-12)


@pytest.mark.parametrize('depth',[30,100,1500])
@pytest.mark.parametrize('left',[False,True])
def test_native_deep_chain_uses_compact_graph(tmp_path,depth,left):
    path=tmp_path/'deep.json';path.write_text(dumps_model_json(document(depth,left=left)))
    model=CatBoost().load_model(path,format='json')
    x=np.array([[-1],[.5],[3],[np.nan]],np.float32)
    np.testing.assert_array_equal(model.predict(x,task_type='GPU'),model.predict(x))


@pytest.mark.parametrize('dimensions',[2,7,64])
def test_native_non_symmetric_vector_bias_scale_and_ranges(tmp_path,dimensions):
    left=np.linspace(-.1,.7,dimensions);right=np.linspace(.3,-.5,dimensions)
    leaf=lambda values:{'value':list(values),'weight':2}
    spec={'features_info':{'float_features':[{'feature_index':0,'flat_feature_index':0,'feature_id':'x',
            'borders':[.5],'has_nans':False,'nan_value_treatment':'AsIs'}]},
          'scale_and_bias':[1.2718281828459,list(np.linspace(.2,-.4,dimensions))],
          'trees':[{'split':{'split_type':'FloatFeature','float_feature_index':0,'border':.5,'split_index':0},
                    'left':leaf(left),'right':leaf(right)},leaf(right-left)]}
    path=tmp_path/'vector.json';path.write_text(json.dumps(spec));model=CatBoost().load_model(path,format='json')
    x=np.array([[-1],[.5],[2]],np.float32)
    for begin,end in ((0,0),(1,2)):
        np.testing.assert_allclose(model.predict(x,task_type='GPU',ntree_start=begin,ntree_end=end),
            model.predict(x,ntree_start=begin,ntree_end=end),rtol=2e-12,atol=2e-12)
