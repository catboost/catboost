"""Overflow-safe positive prefix scans through the full uint32 pair-ID domain."""
import ctypes as ct
import hashlib,platform,subprocess
from pathlib import Path
import numpy as np
import pytest

@pytest.fixture(scope='module')
def prefix():
    if platform.system()!='Darwin' or platform.machine()!='arm64':pytest.skip('Requires Apple GPU')
    source=Path(__file__).with_name('query_prefix_probe.mm');root=source.parents[1];header=root/'native/metal_query_sampler_kernels.h'
    digest=hashlib.sha256(source.read_bytes()+header.read_bytes()).hexdigest()[:20];output=root/'.build'/f'query_prefix_{digest}.dylib'
    if not output.exists():
        subprocess.run(['xcrun','clang++','-std=c++17','-O2','-fobjc-arc','-dynamiclib','-framework','Foundation','-framework','Metal',str(source),'-o',str(output)],check=True,capture_output=True,text=True)
    lib=ct.CDLL(str(output));lib.cbm_query_prefix_probe.argtypes=[ct.c_void_p,ct.c_uint32,ct.c_uint32,ct.c_void_p,ct.c_void_p,ct.c_uint32];lib.cbm_query_prefix_probe.restype=ct.c_int
    def run(values,limit):
        values=np.ascontiguousarray(values,np.uint32);actual=np.empty_like(values);error=ct.create_string_buffer(4096)
        code=lib.cbm_query_prefix_probe(values.ctypes.data,len(values),limit,actual.ctypes.data,error,len(error))
        if code:raise ValueError(error.value.decode())
        return actual
    return run

@pytest.mark.parametrize('limit',[0,1,2**24,2**24+1,2**31-1,2**32-2])
@pytest.mark.parametrize('rows',[1,2,31,32,33,255,256,257,65537])
def test_prefix_never_wraps_at_simd_block_or_recursive_boundaries(prefix,limit,rows):
    rng=np.random.default_rng(rows+limit);values=rng.integers(0,limit+2,size=rows,dtype=np.uint32)
    actual=prefix(values,limit);expected=np.minimum(np.cumsum(values,dtype=np.uint64),limit+1).astype(np.uint32)
    np.testing.assert_array_equal(actual,expected)

@pytest.mark.parametrize('limit',[2**24+1,2**31-1,2**32-2])
def test_long_scan_preserves_small_exact_prefixes_before_saturating(prefix,limit):
    values=np.zeros(65537,np.uint32);values[::31]=101;values[-2]=limit
    expected=np.minimum(np.cumsum(values,dtype=np.uint64),limit+1).astype(np.uint32)
    np.testing.assert_array_equal(prefix(values,limit),expected)


def test_reserved_uint32_sentinel_cannot_be_used_as_limit(prefix):
    with pytest.raises(ValueError,match='bounds'):prefix([1,1],2**32-1)
