"""Resident vector variable-tree GPU inference and counted-buffer contracts."""
import copy
import ctypes as ct
from types import SimpleNamespace
import numpy as np
import pytest
from catboost_metal import _greedy,_multiclass
from catboost_metal._greedy_inference import EvaluationCursor,predict_bins,tree_depth
from test_greedy_vector_training import problem,route,OBJECTIVES
from test_greedy_training import POLICIES

@pytest.fixture(autouse=True)
def no_cpu_fit(monkeypatch):
    from catboost import CatBoost
    def forbidden(*a,**k):raise AssertionError('CPU CatBoost fitting is forbidden')
    monkeypatch.setattr(CatBoost,'_fit',forbidden)


@pytest.mark.parametrize('objective',OBJECTIVES)
@pytest.mark.parametrize('policy',POLICIES)
@pytest.mark.parametrize('classes',[3,7,64])
def test_resident_vector_cursor_matches_independent_routing_and_training(objective,policy,classes):
    args,banks,initial=problem(objective,policy,classes=classes,iterations=3)
    with _multiclass.Session(**args) as training, EvaluationCursor(banks[0],initial_predictions=initial[0]) as cursor:
        expected=initial[0].copy();trees=[]
        for _ in range(3):
            tree=training.step();trees.append(tree)
            expected=np.float32(expected+tree.leaf_values[route(tree,banks[0])])
            np.testing.assert_array_equal(cursor.add_tree(tree),expected)
            np.testing.assert_allclose(cursor.predictions(),training.predictions(),rtol=5e-6,atol=6e-7)
        stats=cursor.stats
        assert stats['dataset_uploads']==1 and stats['bins_upload_bytes']==banks[0].size
        assert stats['tree_upload_bytes']==sum(t.nodes.nbytes+t.leaf_values.nbytes for t in trees)
        assert stats['resident_bytes']==banks[0].nbytes+initial[0].nbytes+trees[-1].nodes.nbytes+trees[-1].leaf_values.nbytes
        assert stats['kernel_dispatches']==3
        independent=np.zeros_like(initial[0])
        for tree in trees:independent+=tree.leaf_values[route(tree,banks[0])]
        np.testing.assert_array_equal(predict_bins(banks[0],trees),independent)
    assert cursor.stats==stats
    with pytest.raises(RuntimeError,match='closed'):cursor.predictions()


@pytest.mark.parametrize('dimensions',[2,3,7,64])
@pytest.mark.parametrize('rows',[0,1,257])
def test_vector_bias_root_trees_ranges_and_empty_forests(dimensions,rows):
    bins=np.zeros((2,rows),np.uint8);bias=np.linspace(-1,1,dimensions,dtype=np.float32)
    values=np.linspace(.1,.3,dimensions,dtype=np.float32)
    tree=SimpleNamespace(nodes=np.zeros((1,6),np.uint32),leaf_values=values[None],leaf_weights=np.ones(1,np.float32))
    baseline=np.broadcast_to(bias,(rows,dimensions)).copy()
    np.testing.assert_array_equal(predict_bins(bins,[],bias),baseline)
    np.testing.assert_array_equal(predict_bins(bins,[tree,tree],bias,tree_end=0),baseline)
    expected=baseline.copy();expected+=values;expected+=values
    np.testing.assert_array_equal(predict_bins(bins,[tree,tree],bias),expected)
    np.testing.assert_array_equal(predict_bins(bins,[tree,tree],bias,tree_start=1),np.broadcast_to(values,(rows,dimensions)))
    np.testing.assert_array_equal(predict_bins(bins,[tree],bias,tree_start=1,tree_end=1),np.zeros((rows,dimensions),np.float32))


@pytest.mark.parametrize('dimensions',[2,3,64])
@pytest.mark.parametrize('corruption',['shape','leaf','cycle','nan','feature','shared'])
def test_invalid_vector_tree_rejected_without_advancing_cursor(dimensions,corruption):
    nodes=np.array([[0,0,0,1,2,2**32-1],[0,0,0,0,0,0],[0,0,0,0,0,1]],np.uint32)
    tree=SimpleNamespace(nodes=nodes,leaf_values=np.ones((2,dimensions),np.float32))
    with EvaluationCursor(np.zeros((1,5),np.uint8),dimensions=dimensions) as cursor:
        broken=copy.deepcopy(tree)
        if corruption=='shape':broken.leaf_values=broken.leaf_values[:,:-1]
        elif corruption=='leaf':broken.nodes[2,5]=2
        elif corruption=='cycle':broken.nodes[0,3]=0
        elif corruption=='shared':broken.nodes[0,4]=1
        elif corruption=='feature':broken.nodes[0,0]=1
        elif corruption=='nan':broken.leaf_values[0,-1]=np.nan
        before=cursor.predictions();stats=cursor.stats
        with pytest.raises(ValueError):cursor.add_tree(broken)
        np.testing.assert_array_equal(cursor.predictions(),before);assert cursor.stats==stats
        np.testing.assert_array_equal(cursor.add_tree(tree),np.ones((5,dimensions),np.float32))


@pytest.mark.parametrize('dimensions',[2,3,64])
def test_counted_c_api_rejects_truncated_values_and_wrong_scalar_entrypoint(dimensions):
    with EvaluationCursor(np.zeros((1,5),np.uint8),dimensions=dimensions) as cursor:
        nodes=np.zeros((1,6),np.uint32);values=np.ones(dimensions,np.float32);output=np.zeros((5,dimensions),np.float32)
        n=nodes.ctypes.data_as(ct.POINTER(_greedy.Node));v=values.ctypes.data_as(ct.POINTER(ct.c_float));o=output.ctypes.data_as(ct.POINTER(ct.c_float))
        error=ct.create_string_buffer(2048);lib=cursor._lib;before=cursor.stats
        for value_count,prediction_count in [(dimensions-1,output.size),(dimensions,output.size-1),(dimensions*65537,output.size)]:
            code=lib.cbm_greedy_evaluation_add_vector_tree(cursor._handle,n,1,v,value_count,o,prediction_count,error,len(error))
            assert code and error.value
            np.testing.assert_array_equal(cursor.predictions(),0);assert cursor.stats==before
        code=lib.cbm_greedy_evaluation_add_tree(cursor._handle,n,1,v,1,o,output.size,error,len(error))
        assert code and b'add_vector_tree' in error.value
        assert not lib.cbm_greedy_evaluation_add_vector_tree(cursor._handle,n,1,v,dimensions,o,output.size,error,len(error))
        np.testing.assert_array_equal(output,1)


@pytest.mark.parametrize('options',[dict(dimensions=0),dict(dimensions=65),dict(dimensions=True),
    dict(bias=[1,2],dimensions=3),dict(initial_predictions=np.zeros((5,3)),dimensions=2),
    dict(initial_predictions=np.zeros((4,3))),dict(bias=[0,np.inf]),dict(bias=[[1,2]])])
def test_invalid_vector_evaluation_dimensions_rejected_before_library_load(monkeypatch,options):
    monkeypatch.setattr(_greedy,'build_library',lambda:pytest.fail('Invalid dimensions reached library load'))
    with pytest.raises(ValueError):EvaluationCursor(np.zeros((1,5),np.uint8),**options)
