#pragma once

// Append after metal_bootstrap_kernels.h. CUDA's additive weak-target formula
// and MWC normal distribution are preserved. Device seed expansion retains
// Metal's documented per-item/absolute-iteration adaptation; a separate domain
// per weak call prevents reusing bootstrap or structure-score samples.
static const char* CBMMetalLangevinSource = R"METAL(
struct LangevinWeakParams {
    BootstrapParams random;
    uint offset, stride, filter_bootstrap, reserved;
};
kernel void AddLangevinWeakNoise(device float* values [[buffer(0)]],
    const device float* multipliers [[buffer(1)]], constant LangevinWeakParams& p [[buffer(2)]],
    uint index [[thread_position_in_grid]]) {
    if (index >= p.random.rows || p.random.noise_scale == 0.0f) return;
    if (p.filter_bootstrap && multipliers[index] == 0.0f) return;
    ulong seed = BootstrapSeedForItem(index, p.random);
    values[p.offset + index * p.stride] += p.random.noise_scale * BootstrapNextNormal(seed);
}
)METAL";
