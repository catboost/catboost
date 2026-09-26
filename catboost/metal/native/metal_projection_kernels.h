#pragma once

static const char* CBMProjectionMetalSource = R"METAL(
#include <metal_stdlib>
using namespace metal;

struct ProjectionParams { uint rows, cats, bins, components, hasPermutation; };

kernel void ProjectionHash(
    device const uint* cats [[buffer(0)]], device const uchar* bins [[buffer(1)]],
    device const uchar* types [[buffer(2)]], device const uint* features [[buffer(3)]],
    device const uint* thresholds [[buffer(4)]], device const uint* permutation [[buffer(5)]],
    device ulong* hashes [[buffer(6)]], device uint* lowKeys [[buffer(7)]],
    device uint* rows [[buffer(8)]], constant ProjectionParams& p [[buffer(9)]],
    uint rank [[thread_position_in_grid]]) {
    if (rank >= p.rows) return;
    const uint row = p.hasPermutation ? permutation[rank] : rank;
    const ulong multiplier = 0x4906ba494954cb65ul;
    ulong hash = 0;
    for (uint component = 0; component < p.components; ++component) {
        const uint feature = features[component];
        ulong value;
        if (types[component] == 0) {
            value = ulong(long(as_type<int>(cats[ulong(feature) * p.rows + row])));
        } else {
            const uint bin = bins[ulong(feature) * p.rows + row];
            value = types[component] == 1 ? ulong(bin > thresholds[component])
                                         : ulong(bin == thresholds[component]);
        }
        hash = multiplier * (hash + multiplier * value);
    }
    hashes[row] = hash;
    lowKeys[rank] = uint(hash);
    rows[rank] = row;
}

kernel void ProjectionHighKeys(device const ulong* hashes [[buffer(0)]],
    device const uint* rows [[buffer(1)]], device uint* highKeys [[buffer(2)]],
    constant uint& count [[buffer(3)]], uint index [[thread_position_in_grid]]) {
    if (index < count) highKeys[index] = uint(hashes[rows[index]] >> 32);
}

kernel void ProjectionGatherHashes(device const ulong* hashes [[buffer(0)]],
    device const uint* rows [[buffer(1)]], device ulong* sortedHashes [[buffer(2)]],
    constant uint& count [[buffer(3)]], uint index [[thread_position_in_grid]]) {
    if (index < count) sortedHashes[index] = hashes[rows[index]];
}
)METAL";
