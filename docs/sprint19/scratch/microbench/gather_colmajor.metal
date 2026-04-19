// gather_colmajor.metal
// Sprint 19 / S19-01c micro-benchmark — column-major compressedIndex gather.
//
// PURPOSE
//   Identical shape to gather_rowmajor.metal but reads column-major layout:
//   compressedIndex[featureColumnIdx * totalNumDocs + docIdx].
//   This is the DEC-015 address expression. For sorted docIdx (monotone
//   increasing), 32 consecutive threads access 32 consecutive 4-byte slots =
//   128 bytes = 1 cache line. The S19-01b model predicted this would reduce
//   stall rounds from 4 to 1 per 32-doc batch.
//
// LAYOUT
//   compressedIndex: [lineSize * totalNumDocs] uint32   (column-major)
//     compressedIndex[col * totalNumDocs + doc]
//   docIndices: [totalNumDocs] uint32  (same sorted indices as row-major bench)
//   output: [totalNumDocs] uint32
//
// GRID — identical to gather_rowmajor.metal.
//
// NOT COMPILED into any .metallib — scratch only.

#include <metal_stdlib>
using namespace metal;

kernel void gather_colmajor(
    device const uint*  compressedIndex [[buffer(0)]],   // [lineSize * totalNumDocs]
    device const uint*  docIndices      [[buffer(1)]],   // [totalNumDocs] sorted
    device const uint&  lineSize        [[buffer(2)]],   // kept for signature symmetry
    device const uint&  featureColumnIdx [[buffer(3)]],
    device const uint&  totalNumDocs    [[buffer(4)]],
    device uint*        output          [[buffer(5)]],
    uint                gid             [[thread_position_in_grid]])
{
    if (gid >= totalNumDocs) return;

    const uint docIdx = docIndices[gid];
    // Column-major: stride = 4 bytes/doc — all 32 lanes in a SIMD group
    // with consecutive gid values access a contiguous 128-byte cache line.
    output[gid] = compressedIndex[featureColumnIdx * totalNumDocs + docIdx];
}
