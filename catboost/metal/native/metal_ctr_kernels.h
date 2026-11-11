#pragma once

// Translates ctr_calcers.h's segmented exclusive histories and
// ctr_calcers.cu::FillBinarizedTargetsStats / NonWeightedBinFreqCtrsImpl.
static const char* CBMCtrMetalSource = R"METAL(
#include <metal_stdlib>
using namespace metal;

struct CtrParams {
    uint rows, categories, type, border;
    float priorNumerator, priorDenominator;
    uint offset;
};

kernel void CtrInitialize(device const uint* indices [[buffer(0)]],
                          device const float* targets [[buffer(1)]],
                          device float2* values [[buffer(2)]],
                          constant CtrParams& p [[buffer(3)]],
                          uint i [[thread_position_in_grid]]) {
    if (i >= p.rows) return;
    float target = targets[indices[i]];
    float value = p.type == 0 ? float(target > float(p.border))
                : p.type == 1 ? float(target == float(p.border))
                : p.type == 2 ? target : 1.0f;
    values[i] = float2(value, 1.0f);
}

kernel void CtrSegmentedScan(device const uint* categories [[buffer(0)]],
                             device const float2* source [[buffer(1)]],
                             device float2* destination [[buffer(2)]],
                             constant CtrParams& p [[buffer(3)]],
                             uint i [[thread_position_in_grid]]) {
    if (i >= p.rows) return;
    float2 value = source[i];
    if (i >= p.offset && categories[i - p.offset] == categories[i])
        value += source[i - p.offset];
    destination[i] = value;
}

kernel void CtrWriteHistory(device const uint* categories [[buffer(0)]],
                            device const uint* indices [[buffer(1)]],
                            device const float2* inclusive [[buffer(2)]],
                            device float* values [[buffer(3)]],
                            device float* sums [[buffer(4)]],
                            device uint* counts [[buffer(5)]],
                            constant CtrParams& p [[buffer(6)]],
                            uint i [[thread_position_in_grid]]) {
    if (i >= p.rows) return;
    uint category = categories[i];
    // Read the previous inclusive value, rather than subtracting this row's
    // target, so its target never enters its own encoding even numerically.
    float2 prefix = i > 0 && categories[i - 1] == category
                  ? inclusive[i - 1] : float2(0.0f);
    values[indices[i]] = (prefix.x + p.priorNumerator) / (prefix.y + p.priorDenominator);
    if (i + 1 == p.rows || categories[i + 1] != category) {
        sums[category] = inclusive[i].x;
        counts[category] = uint(inclusive[i].y);
    }
}

kernel void CtrInitializeGroupHeads(device const uint* categories [[buffer(0)]],
                                    device const uint* indices [[buffer(1)]],
                                    device const uint* groupIds [[buffer(2)]],
                                    device uint* heads [[buffer(3)]],
                                    constant CtrParams& p [[buffer(4)]],
                                    uint i [[thread_position_in_grid]]) {
    if (i >= p.rows) return;
    bool begins = i == 0 || categories[i - 1] != categories[i]
        || groupIds[indices[i - 1]] != groupIds[indices[i]];
    heads[i] = begins ? i : 0;
}

kernel void CtrScanWithGroupHeads(device const uint* categories [[buffer(0)]],
                                  device const float2* source [[buffer(1)]],
                                  device float2* destination [[buffer(2)]],
                                  device const uint* sourceHeads [[buffer(3)]],
                                  device uint* destinationHeads [[buffer(4)]],
                                  constant CtrParams& p [[buffer(5)]],
                                  uint i [[thread_position_in_grid]]) {
    if (i >= p.rows) return;
    float2 value = source[i];
    uint head = sourceHeads[i];
    if (i >= p.offset) {
        if (categories[i - p.offset] == categories[i]) value += source[i - p.offset];
        head = max(head, sourceHeads[i - p.offset]);
    }
    destination[i] = value;
    destinationHeads[i] = head;
}

kernel void CtrWriteGroupedHistory(device const uint* categories [[buffer(0)]],
                                   device const uint* indices [[buffer(1)]],
                                   device const float2* inclusive [[buffer(2)]],
                                   device const uint* heads [[buffer(3)]],
                                   device float* values [[buffer(4)]],
                                   device float* sums [[buffer(5)]],
                                   device uint* counts [[buffer(6)]],
                                   constant CtrParams& p [[buffer(7)]],
                                   uint i [[thread_position_in_grid]]) {
    if (i >= p.rows) return;
    uint category = categories[i];
    uint head = heads[i];
    // The fetched prefix never contains this group's targets. Subtracting
    // its sum from inclusive[i] would introduce cancellation and leakage.
    float2 prefix = head > 0 && categories[head - 1] == category
                  ? inclusive[head - 1] : float2(0.0f);
    values[indices[i]] = (prefix.x + p.priorNumerator) / (prefix.y + p.priorDenominator);
    if (i + 1 == p.rows || categories[i + 1] != category) {
        sums[category] = inclusive[i].x;
        counts[category] = uint(inclusive[i].y);
    }
}

kernel void CtrWriteFrequency(device const uint* categories [[buffer(0)]],
                              device const uint* indices [[buffer(1)]],
                              device const uint* counts [[buffer(2)]],
                              device float* values [[buffer(3)]],
                              constant CtrParams& p [[buffer(4)]],
                              uint i [[thread_position_in_grid]]) {
    if (i >= p.rows) return;
    values[indices[i]] = (float(counts[categories[i]]) + p.priorNumerator)
                      / (float(p.rows) + p.priorDenominator);
}
)METAL";
