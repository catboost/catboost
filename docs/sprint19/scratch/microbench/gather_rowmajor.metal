// gather_rowmajor.metal
// Sprint 19 / S19-01c micro-benchmark — row-major compressedIndex gather.
//
// PURPOSE
//   Minimal kernel that isolates the compressedIndex gather cost in row-major
//   (doc-major) layout: compressedIndex[docIdx * lineSize + featureColumnIdx].
//   No histogram accumulation, no simd_shuffle, no TG memory.
//   One uint32 read per thread, result written to output buffer.
//
// LAYOUT
//   compressedIndex: [totalNumDocs * lineSize] uint32   (row-major)
//     compressedIndex[doc * lineSize + col]
//   docIndices: [totalNumDocs] uint32  (sorted partition indices)
//   output: [totalNumDocs] uint32
//
// GRID (matching production L1a kernel at N=50k, lineSize=25)
//   threadgroup size: 256
//   threadgroups:     ceil(totalNumDocs / 256)
//   Each thread reads one doc's column value and writes it.
//
// NOTE: The production kernel processes docs in sorted (partition) order
//   using docIndices[sortedPos]. We replicate that here: thread t reads
//   docIndices[t] and then gathers compressedIndex[docIdx * lineSize + col].
//   This is the exact access pattern that produces 25 cache-line misses per
//   32-lane SIMD group per batch in the production kernel.
//
// NOT COMPILED into any .metallib — scratch only.

#include <metal_stdlib>
using namespace metal;

kernel void gather_rowmajor(
    device const uint*  compressedIndex [[buffer(0)]],   // [totalNumDocs * lineSize]
    device const uint*  docIndices      [[buffer(1)]],   // [totalNumDocs] sorted
    device const uint&  lineSize        [[buffer(2)]],
    device const uint&  featureColumnIdx [[buffer(3)]],
    device const uint&  totalNumDocs    [[buffer(4)]],
    device uint*        output          [[buffer(5)]],
    uint                gid             [[thread_position_in_grid]])
{
    if (gid >= totalNumDocs) return;

    // Replicate production access: fetch docIdx from sorted partition indices,
    // then gather compressedIndex[docIdx * lineSize + featureColumnIdx].
    const uint docIdx = docIndices[gid];
    // Row-major: stride = lineSize (100 bytes at gate lineSize=25)
    output[gid] = compressedIndex[docIdx * lineSize + featureColumnIdx];
}
