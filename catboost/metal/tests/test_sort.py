"""Validate unsigned ordering and stability of the real Metal radix sort."""

import ctypes as ct
from pathlib import Path
import platform
import subprocess

import numpy as np
import pytest

from catboost_metal._sort import stable_sort, _build_library, _load, _Stats


METAL = pytest.mark.skipif(platform.system() != "Darwin" or platform.machine() != "arm64",
                          reason="GPU radix-sort checks require Apple Silicon")


@METAL
@pytest.mark.parametrize("rows", [0, 1, 31, 32, 33, 255, 256, 257, 1031, 4096, 4097, 65539, 1048579])
def test_unsigned_sort_matches_stable_reference(rows):
    rng = np.random.default_rng(919)
    keys = rng.integers(0, 2 ** 32, rows, dtype=np.uint32)
    if rows > 3:
        keys[::7] = 0xFFFFFFFF
        keys[1::7] = 0
        keys[2::7] = 0x80000000
    expected = np.argsort(keys, kind="stable")
    original = keys.copy()
    ordered, permutation, stats = stable_sort(keys)
    np.testing.assert_array_equal(ordered, keys[expected])
    np.testing.assert_array_equal(permutation, expected.astype(np.uint32))
    np.testing.assert_array_equal(keys, original)
    assert stats["backend"] == "Metal"
    assert stats["radix_passes"] == (8 if rows else 0)
    assert stats["kernel_dispatches"] >= (24 if rows else 0)


@METAL
@pytest.mark.parametrize("kind", ["equal", "ascending", "descending", "digit_adversarial"])
def test_stability_with_arbitrary_payload_values(kind):
    rows = 65539
    rng = np.random.default_rng(720)
    if kind == "equal":
        keys = np.full(rows, 0xFFFFFFFF, dtype=np.uint32)
    elif kind == "ascending":
        keys = np.arange(rows, dtype=np.uint32) // 7
    elif kind == "descending":
        keys = (np.arange(rows, dtype=np.uint32) // 7)[::-1]
    else:
        # Every hex digit matters. Adjacent keys differ in distant digits and
        # repeat across many groups; signed comparisons or unstable passes fail.
        keys = np.resize(np.array([0xF000000F, 0x1000000F, 0xF0000010,
                                   0x10000000, 0x80000000, 0, 0xFFFFFFFF], np.uint32), rows)
    payload = rng.integers(0, 2 ** 32, rows, dtype=np.uint32)
    expected = np.argsort(keys, kind="stable")
    sorted_keys, sorted_payload, _ = stable_sort(keys, payload)
    np.testing.assert_array_equal(sorted_keys, keys[expected])
    np.testing.assert_array_equal(sorted_payload, payload[expected])


@METAL
def test_deterministic_repeated_sort_and_noncontiguous_input():
    keys = np.arange(5000, dtype=np.uint32)[::-3] % 31
    payload = np.arange(len(keys), dtype=np.uint32)[::-1]
    expected = np.argsort(keys, kind="stable")
    for _ in range(3):
        actual_keys, actual_payload, _ = stable_sort(keys, payload)
        np.testing.assert_array_equal(actual_keys, keys[expected])
        np.testing.assert_array_equal(actual_payload, payload[expected])


@METAL
def test_sort_does_not_use_host_sorting(monkeypatch):
    keys = np.array([5, 1, 5, 0xFFFFFFFF, 0, 1], dtype=np.uint32)
    def forbidden(*args, **kwargs):
        raise AssertionError("The GPU sort must not call a host sorting algorithm")
    monkeypatch.setattr(np, "sort", forbidden)
    monkeypatch.setattr(np, "argsort", forbidden)
    monkeypatch.setattr(np, "lexsort", forbidden)
    actual, payload, _ = stable_sort(keys)
    np.testing.assert_array_equal(actual, [0, 1, 1, 5, 5, 0xFFFFFFFF])
    np.testing.assert_array_equal(payload, [4, 1, 5, 0, 2, 3])


@pytest.mark.parametrize("keys,payload", [
    ([-1], None), ([2 ** 32], None), ([1.5], None), ([[1]], None), ([True], None),
    ([1], [-1]), ([1], [2 ** 32]), ([1], [1.0]), ([1], []), ([1], [[1]]),
])
def test_invalid_keys_and_payload_are_rejected(keys, payload):
    with pytest.raises(ValueError):
        stable_sort(keys, payload)


@METAL
def test_native_inplace_arrays_and_overlap_validation():
    keys = np.array([5, 2, 5, 0xFFFFFFFF, 0], dtype=np.uint32)
    payload = np.array([99, 3, 42, 17, 100], dtype=np.uint32)
    library = _load(_build_library())
    u32 = lambda array: array.ctypes.data_as(ct.POINTER(ct.c_uint32))
    stats, error = _Stats(), ct.create_string_buffer(2048)
    assert library.cbm_sort_u32(u32(keys), u32(payload), len(keys), u32(keys), u32(payload),
                                ct.byref(stats), error, len(error)) == 0
    np.testing.assert_array_equal(keys, [0, 2, 5, 5, 0xFFFFFFFF])
    np.testing.assert_array_equal(payload, [100, 3, 99, 42, 17])
    assert library.cbm_sort_u32(u32(keys), u32(payload), len(keys), u32(keys), u32(keys),
                                ct.byref(stats), error, len(error)) != 0
    assert b"overlap" in error.value


@METAL
def test_resident_two_stage_sort_without_intermediate_readback(tmp_path):
    # Exact leaf estimation needs two stable sorts in one resident pipeline:
    # residual order first, then leaf order while preserving residual order.
    # Private buffers and a GPU-only key gather make a host fallback impossible.
    native = Path(__file__).resolve().parents[1] / "native"
    source = tmp_path / "resident_sort.mm"
    source.write_text(r'''
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "metal_sort.h"
#include <cstring>
#include <exception>
extern "C" int resident_sort(const uint32_t* residuals, const uint32_t* leaves,
                              uint32_t count, uint32_t* result, uint64_t* dispatches) {
    @autoreleasepool {
        try {
            id<MTLDevice> device = MTLCreateSystemDefaultDevice();
            id<MTLCommandQueue> queue = [device newCommandQueue];
            id<MTLCommandBuffer> command = [queue commandBuffer];
            const size_t bytes = size_t(count) * 4;
            auto input = [device newBufferWithBytes:residuals length:bytes options:MTLResourceStorageModeShared];
            auto leaf = [device newBufferWithBytes:leaves length:bytes options:MTLResourceStorageModeShared];
            auto keys = [device newBufferWithLength:bytes options:MTLResourceStorageModePrivate];
            auto payload = [device newBufferWithLength:bytes options:MTLResourceStorageModePrivate];
            auto output = [device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
            auto blit = [command blitCommandEncoder];
            [blit copyFromBuffer:input sourceOffset:0 toBuffer:keys destinationOffset:0 size:bytes];
            [blit endEncoding];
            NSError* error = nil;
            auto library = [device newLibraryWithSource:@R"(
#include <metal_stdlib>
using namespace metal;
kernel void ids(device uint* values [[buffer(0)]], uint row [[thread_position_in_grid]]) {
    values[row] = row;
}
kernel void gather(const device uint* ids [[buffer(0)]], const device uint* leaves [[buffer(1)]],
                   device uint* keys [[buffer(2)]], uint row [[thread_position_in_grid]]) {
    keys[row] = leaves[ids[row]];
})" options:nil error:&error];
            if (!library) return 2;
            auto initialize = [device newComputePipelineStateWithFunction:[library newFunctionWithName:@"ids"] error:&error];
            auto gather = [device newComputePipelineStateWithFunction:[library newFunctionWithName:@"gather"] error:&error];
            auto encoder = [command computeCommandEncoder];
            [encoder setComputePipelineState:initialize];
            [encoder setBuffer:payload offset:0 atIndex:0];
            [encoder dispatchThreads:MTLSizeMake(count,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
            [encoder endEncoding];
            CBMEncodeSortU32(command, keys, payload, count, keys, payload, dispatches);
            encoder = [command computeCommandEncoder];
            [encoder setComputePipelineState:gather];
            [encoder setBuffer:payload offset:0 atIndex:0];
            [encoder setBuffer:leaf offset:0 atIndex:1];
            [encoder setBuffer:keys offset:0 atIndex:2];
            [encoder dispatchThreads:MTLSizeMake(count,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
            [encoder endEncoding];
            CBMEncodeSortU32(command, keys, payload, count, keys, payload, dispatches);
            blit = [command blitCommandEncoder];
            [blit copyFromBuffer:payload sourceOffset:0 toBuffer:output destinationOffset:0 size:bytes];
            [blit endEncoding];
            [command commit];
            [command waitUntilCompleted];
            if (command.status != MTLCommandBufferStatusCompleted) return 3;
            std::memcpy(result, output.contents, bytes);
            return 0;
        } catch (const std::exception&) {
            return 1;
        }
    }
}
''')
    path = tmp_path / "resident_sort.dylib"
    subprocess.run(["xcrun", "clang++", "-std=c++17", "-O2", "-fobjc-arc", "-dynamiclib",
                    "-framework", "Foundation", "-framework", "Metal", "-I", str(native),
                    str(native / "metal_sort.mm"), str(source), "-o", str(path)],
                   check=True, capture_output=True, text=True)
    library = ct.CDLL(str(path))
    u32 = ct.POINTER(ct.c_uint32)
    library.resident_sort.argtypes = [u32, u32, ct.c_uint32, u32, ct.POINTER(ct.c_uint64)]
    library.resident_sort.restype = ct.c_int
    rng = np.random.default_rng(219)
    residuals = rng.integers(0, 101, 65539, dtype=np.uint32)
    leaves = rng.integers(0, 17, len(residuals), dtype=np.uint32)
    expected = np.lexsort((np.arange(len(residuals)), residuals, leaves))
    result = np.empty_like(residuals)
    dispatches = ct.c_uint64()
    ptr = lambda values: values.ctypes.data_as(u32)
    assert library.resident_sort(ptr(residuals), ptr(leaves), len(residuals), ptr(result),
                                  ct.byref(dispatches)) == 0
    assert dispatches.value >= 48
    np.testing.assert_array_equal(result, expected.astype(np.uint32))
