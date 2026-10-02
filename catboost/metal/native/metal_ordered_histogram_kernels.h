#pragma once
#include <cstdint>

// Fold slots are padded to a power of two. A partition key is
// leaf*FoldSlots+fold, so the shared deep radix partition appends one leaf bit
// while retaining fold membership. Row payloads index PACKED occurrences;
// metadata packs original row in low31 bits and estimate membership in bit31.
struct CBMOrderedHistogramParams {
    uint32_t Rows, PackedRows, FoldCount, FoldSlots;
    uint32_t Partitions, Features, FeatureBegin, TotalBins;
    uint32_t JobCapacity, TileRows, Reuse, DerivativeOffset;
    uint32_t Candidates, LeafMask, Reserved0, Reserved1;
};
static_assert(sizeof(CBMOrderedHistogramParams) == 64);

// Append after shared KernelParams, objective helpers, deep partitions and
// Ordered foundation. Histogram cells carry a float4 expansion (high+low),
// preserving all four estimate/quality weight/gradient channels through local
// collisions, tile merges, prefix sums and sibling subtraction. Numeric spans
// include one overflow bucket, so their last prefix is the complete partition.
static const char* CBMMetalOrderedHistogramSource = R"METAL(
struct OrderedHistogramParams {
    uint rows, packed_rows, fold_count, fold_slots;
    uint partitions, features, feature_begin, total_bins;
    uint job_capacity, tile_rows, reuse, derivative_offset;
    uint candidates, leaf_mask, reserved0, reserved1;
};

inline void OrderedHistogramMerge(thread float4& high, thread float4& low, float4 value) {
    const float4 sum = high + value;
    const float4 part = sum - high;
    const float4 error = (high - (sum - part)) + (value - part);
    const float4 tail = low + error;
    const float4 next = sum + tail;
    const float4 next_part = next - sum;
    low = (sum - (next - next_part)) + (tail - next_part);
    high = next;
}

inline void OrderedHistogramAtomic(threadgroup atomic_uint* high, threadgroup atomic_uint* low, float value) {
    if (value == 0) return;
    uint previous = atomic_load_explicit(high, memory_order_relaxed);
    float error;
    while (true) {
        const float old = as_type<float>(previous), sum = old + value, part = sum - old;
        error = (old - (sum - part)) + (value - part);
        if (atomic_compare_exchange_weak_explicit(high, &previous, as_type<uint>(sum),
                memory_order_relaxed, memory_order_relaxed)) break;
    }
    if (error != 0) {
        previous = atomic_load_explicit(low, memory_order_relaxed);
        while (!atomic_compare_exchange_weak_explicit(low, &previous,
            as_type<uint>(as_type<float>(previous) + error), memory_order_relaxed, memory_order_relaxed)) {}
    }
}
inline void OrderedHistogramAtomic(device atomic_uint* high, device atomic_uint* low, float value) {
    if (value == 0) return;
    uint previous = atomic_load_explicit(high, memory_order_relaxed);
    float error;
    while (true) {
        const float old = as_type<float>(previous), sum = old + value, part = sum - old;
        error = (old - (sum - part)) + (value - part);
        if (atomic_compare_exchange_weak_explicit(high, &previous, as_type<uint>(sum),
                memory_order_relaxed, memory_order_relaxed)) break;
    }
    if (error != 0) {
        previous = atomic_load_explicit(low, memory_order_relaxed);
        while (!atomic_compare_exchange_weak_explicit(low, &previous,
            as_type<uint>(as_type<float>(previous) + error), memory_order_relaxed, memory_order_relaxed)) {}
    }
}

kernel void InitializeOrderedHistogramOccurrences(const device uint* permutation [[buffer(0)]],
    const device uint4* folds [[buffer(1)]], device uint* metadata [[buffer(2)]],
    device uint* row_indices [[buffer(3)]], device uint* leaf_ids [[buffer(4)]],
    device uint* offsets [[buffer(5)]], constant OrderedHistogramParams& p [[buffer(6)]],
    uint2 tid [[thread_position_in_grid]]) {
    if (tid.y < p.fold_count) {
        const uint4 fold = folds[tid.y];
        const uint begin = fold.z - p.derivative_offset;
        if (tid.x < fold.y) {
            const uint slot = begin + tid.x;
            metadata[slot] = permutation[tid.x] | (tid.x < fold.x ? 0x80000000u : 0u);
            row_indices[slot] = slot;
            leaf_ids[slot] = tid.y;
        }
        if (!tid.x) offsets[tid.y] = begin;
    }
    if (!tid.x && !tid.y) for (uint fold = p.fold_count; fold <= p.fold_slots; ++fold) offsets[fold] = p.packed_rows;
}

kernel void UpdateOrderedHistogramLeafIds(const device uint* metadata [[buffer(0)]],
    const device uint* original_leaf_ids [[buffer(1)]], device uint* leaf_ids [[buffer(2)]],
    constant OrderedHistogramParams& p [[buffer(3)]], uint occurrence [[thread_position_in_grid]]) {
    if (occurrence < p.packed_rows) {
        const uint row = metadata[occurrence] & 0x7fffffffu;
        leaf_ids[occurrence] = (leaf_ids[occurrence] & (p.fold_slots - 1))
            | ((original_leaf_ids[row] & p.leaf_mask) * p.fold_slots);
    }
}

kernel void ResetOrderedHistogramJobs(device atomic_uint* state [[buffer(0)]],
    constant OrderedHistogramParams& p [[buffer(1)]], uint index [[thread_position_in_grid]]) {
    if (!index) for (uint i = 0; i < 4; ++i) atomic_store_explicit(state + i, 0u, memory_order_relaxed);
}
kernel void BuildOrderedHistogramJobs(const device uint* offsets [[buffer(0)]],
    device uint4* jobs [[buffer(1)]], device uint* active [[buffer(2)]], device atomic_uint* state [[buffer(3)]],
    constant OrderedHistogramParams& p [[buffer(4)]], uint partition [[thread_position_in_grid]]) {
    const uint parents = p.reuse ? p.partitions / 2 : p.partitions;
    if (partition >= parents) return;
    uint selected = partition, storage = partition;
    uint size = offsets[partition + 1] - offsets[partition];
    if (p.reuse) {
        const uint right = partition + parents, right_size = offsets[right + 1] - offsets[right];
        if (size + right_size == 0) return;
        selected = size < right_size ? partition : right; // CUDA strict comparison; equal chooses right.
        size = min(size, right_size); storage = right;
    } else if (!size) return;
    active[atomic_fetch_add_explicit(state + 1, 1u, memory_order_relaxed)] = partition;
    if (!size) return;
    const uint count = 1 + (size - 1) / p.tile_rows;
    const uint first = atomic_fetch_add_explicit(state, count, memory_order_relaxed);
    if (first > p.job_capacity || count > p.job_capacity - first) {
        atomic_store_explicit(state + 2, 1u, memory_order_relaxed); return;
    }
    for (uint tile = 0; tile < count; ++tile) {
        const uint begin = offsets[selected] + tile * p.tile_rows;
        jobs[first + tile] = uint4(begin, min(begin + p.tile_rows, offsets[selected + 1]), storage, selected);
    }
}
kernel void OrderedHistogramArguments(const device atomic_uint* state [[buffer(0)]],
    device uint* arguments [[buffer(1)]], constant OrderedHistogramParams& p [[buffer(2)]],
    uint index [[thread_position_in_grid]]) {
    if (index) return;
    const bool valid = atomic_load_explicit(state + 2, memory_order_relaxed) == 0;
    const uint jobs = valid ? atomic_load_explicit(state, memory_order_relaxed) : 0;
    const uint active = valid ? atomic_load_explicit(state + 1, memory_order_relaxed) : 0;
    arguments[0] = jobs; arguments[1] = p.features; arguments[2] = 1;
    arguments[3] = active; arguments[4] = p.features; arguments[5] = 1;
    arguments[6] = p.reuse ? uint((ulong(active) * p.total_bins + 255) / 256) : 0;
    arguments[7] = 1; arguments[8] = 1;
}
kernel void ClearOrderedHistogram(device float4* high [[buffer(0)]], device float4* low [[buffer(1)]],
    constant OrderedHistogramParams& p [[buffer(2)]], uint index [[thread_position_in_grid]]) {
    const ulong cells = ulong(p.reuse ? p.partitions / 2 : p.partitions) * p.total_bins;
    if (ulong(index) < cells) {
        const ulong cell = (p.reuse ? cells : 0) + index;
        high[cell] = 0; low[cell] = 0;
    }
}

// Two collision banks, four statistic channels and two expansion parts use
// 16KiB threadgroup memory. Each packed occurrence is visited once per feature
// per level, or only in its smaller child when parents can be retained.
kernel void ComputeOrderedHistogram(const device uchar* bins [[buffer(0)]],
    const device float2* derivatives [[buffer(1)]], const device uint* metadata [[buffer(2)]],
    const device uint* row_indices [[buffer(3)]], const device uint* feature_offsets [[buffer(4)]],
    const device uint4* jobs [[buffer(5)]], const device atomic_uint* state [[buffer(6)]],
    device atomic_uint* high [[buffer(7)]], device atomic_uint* low [[buffer(8)]],
    constant OrderedHistogramParams& p [[buffer(9)]], uint2 local [[thread_position_in_threadgroup]],
    uint2 group [[threadgroup_position_in_grid]]) {
    const uint tid = local.x;
    if (atomic_load_explicit(state + 2, memory_order_relaxed) ||
        group.x >= atomic_load_explicit(state, memory_order_relaxed) || group.y >= p.features) return;
    const uint begin_bin = feature_offsets[group.y], span = feature_offsets[group.y + 1] - begin_bin;
    if (!span) return;
    const uint4 job = jobs[group.x];
    threadgroup atomic_uint local_high[2048], local_low[2048];
    for (uint cell = tid; cell < 2048; cell += 256) {
        atomic_store_explicit(local_high + cell, 0u, memory_order_relaxed);
        atomic_store_explicit(local_low + cell, 0u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint bank = ((tid >> 5) & 1u) * 1024;
    for (uint position = job.x + tid; position < job.y; position += 256) {
        const uint occurrence = row_indices[position], info = metadata[occurrence], row = info & 0x7fffffffu;
        const uint bin = min(uint(bins[ulong(p.feature_begin + group.y) * p.rows + row]), span - 1);
        const float2 d = derivatives[p.derivative_offset + occurrence];
        const uint channel = (info & 0x80000000u) ? 0 : 2;
        const uint cell = bank + bin * 4 + channel;
        OrderedHistogramAtomic(local_high + cell, local_low + cell, d.y);
        OrderedHistogramAtomic(local_high + cell + 1, local_low + cell + 1, d.x);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < span) {
        float4 h = 0, l = 0;
        for (uint bank_id = 0; bank_id < 2; ++bank_id) {
            float4 bh, bl;
            for (uint component = 0; component < 4; ++component) {
                const uint cell = bank_id * 1024 + tid * 4 + component;
                bh[component] = as_type<float>(atomic_load_explicit(local_high + cell, memory_order_relaxed));
                bl[component] = as_type<float>(atomic_load_explicit(local_low + cell, memory_order_relaxed));
            }
            OrderedHistogramMerge(h, l, bh); OrderedHistogramMerge(h, l, bl);
        }
        const ulong destination = (ulong(job.z) * p.total_bins + begin_bin + tid) * 4;
        for (uint component = 0; component < 4; ++component) {
            OrderedHistogramAtomic(high + destination + component, low + destination + component, h[component]);
            OrderedHistogramAtomic(high + destination + component, low + destination + component, l[component]);
        }
    }
}
kernel void ScanOrderedHistogram(device float4* high [[buffer(0)]], device float4* low [[buffer(1)]],
    const device uint* feature_offsets [[buffer(2)]], const device uint* active [[buffer(3)]],
    constant OrderedHistogramParams& p [[buffer(4)]], uint2 local [[thread_position_in_threadgroup]],
    uint2 group [[threadgroup_position_in_grid]]) {
    const uint tid = local.x;
    const uint begin = feature_offsets[group.y], span = feature_offsets[group.y + 1] - begin;
    if (!span) return;
    const uint storage = active[group.x] + (p.reuse ? p.partitions / 2 : 0);
    const ulong cell = ulong(storage) * p.total_bins + begin + tid;
    threadgroup float4 sh[256], sl[256];
    sh[tid] = tid < span ? high[cell] : float4(0); sl[tid] = tid < span ? low[cell] : float4(0);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = 1; stride < 256; stride <<= 1) {
        const float4 ph = tid >= stride ? sh[tid - stride] : float4(0);
        const float4 pl = tid >= stride ? sl[tid - stride] : float4(0);
        float4 h = sh[tid], l = sl[tid];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        OrderedHistogramMerge(h, l, ph); OrderedHistogramMerge(h, l, pl);
        sh[tid] = h; sl[tid] = l;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid < span) { high[cell] = sh[tid]; low[cell] = sl[tid]; }
}
kernel void SubtractOrderedHistogramSibling(device float4* high [[buffer(0)]], device float4* low [[buffer(1)]],
    const device uint* offsets [[buffer(2)]], const device uint* active [[buffer(3)]],
    const device atomic_uint* state [[buffer(4)]], constant OrderedHistogramParams& p [[buffer(5)]],
    uint index [[thread_position_in_grid]]) {
    if (ulong(index) >= ulong(atomic_load_explicit(state + 1, memory_order_relaxed)) * p.total_bins) return;
    const uint parent = active[index / p.total_bins], bin = index % p.total_bins, right = parent + p.partitions / 2;
    const ulong left_cell = ulong(parent) * p.total_bins + bin, right_cell = ulong(right) * p.total_bins + bin;
    const float4 child_high = high[right_cell], child_low = low[right_cell];
    float4 other_high = high[left_cell], other_low = low[left_cell];
    OrderedHistogramMerge(other_high, other_low, -child_high); OrderedHistogramMerge(other_high, other_low, -child_low);
    const bool left_calculated = offsets[parent + 1] - offsets[parent] < offsets[right + 1] - offsets[right];
    high[left_cell] = left_calculated ? child_high : other_high;
    low[left_cell] = left_calculated ? child_low : other_low;
    high[right_cell] = left_calculated ? other_high : child_high;
    low[right_cell] = left_calculated ? other_low : child_low;
}

// Recover the proven candidate-major foundation ABI. Terminal overflow means
// the last scanned bin is the complete partition, including encoded bin255.
kernel void ExtractOrderedHistogramCandidates(const device float4* high [[buffer(0)]],
    const device float4* low [[buffer(1)]], const device uint* feature_offsets [[buffer(2)]],
    const device uint2* candidates [[buffer(3)]], device float4* statistics [[buffer(4)]],
    constant OrderedHistogramParams& p [[buffer(5)]], uint index [[thread_position_in_grid]]) {
    const uint leaves = p.partitions / p.fold_slots;
    if (ulong(index) >= ulong(p.candidates) * leaves * p.fold_count) return;
    const uint fold = index % p.fold_count, leaf = (index / p.fold_count) % leaves;
    const uint candidate = index / (p.fold_count * leaves);
    const uint2 split = candidates[candidate];
    const uint feature = split.x - p.feature_begin;
    const ulong base = ulong(leaf * p.fold_slots + fold) * p.total_bins;
    const uint border = split.y & 0xffu;
    const ulong cell = base + feature_offsets[feature] + border;
    const ulong last = base + feature_offsets[feature + 1] - 1;
    if (split.y & 0x80000000u) {
        // Equality goes right. Recover that bin from adjacent expanded
        // prefixes, then subtract it from the complete partition for left.
        float4 right_high = high[cell], right_low = low[cell];
        if (border) {
            OrderedHistogramMerge(right_high, right_low, -high[cell - 1]);
            OrderedHistogramMerge(right_high, right_low, -low[cell - 1]);
        }
        float4 left_high = high[last], left_low = low[last];
        OrderedHistogramMerge(left_high, left_low, -right_high);
        OrderedHistogramMerge(left_high, left_low, -right_low);
        statistics[ulong(index) * 2] = left_high + left_low;
        statistics[ulong(index) * 2 + 1] = right_high + right_low;
    } else {
        float4 right_high = high[last], right_low = low[last];
        OrderedHistogramMerge(right_high, right_low, -high[cell]); OrderedHistogramMerge(right_high, right_low, -low[cell]);
        statistics[ulong(index) * 2] = high[cell] + low[cell];
        statistics[ulong(index) * 2 + 1] = right_high + right_low;
    }
}
kernel void ScatterOrderedHistogramScores(const device float2* tile_scores [[buffer(0)]],
    const device uint* candidate_indices [[buffer(1)]], device float2* scores [[buffer(2)]],
    constant OrderedHistogramParams& p [[buffer(3)]], uint candidate [[thread_position_in_grid]]) {
    if (candidate < p.candidates) scores[candidate_indices[candidate]] = tile_scores[candidate];
}

// At depths whose dense per-bin cache cannot fit the workspace cap, retain
// the same packed partitions and reduce candidates only over their own rows.
// This bounded fallback has O(packed_rows*candidates) work, without the old
// extra leaf multiplier. Compact active partitions avoid empty-leaf dispatch.
kernel void ClearOrderedPartitionCandidateStatistics(device float4* statistics [[buffer(0)]],
    constant OrderedHistogramParams& p [[buffer(1)]], uint index [[thread_position_in_grid]]) {
    if (ulong(index) < ulong(p.candidates) * (p.partitions / p.fold_slots) * p.fold_count * 2) statistics[index] = 0;
}
kernel void ComputeOrderedPartitionCandidates(const device uchar* bins [[buffer(0)]],
    const device float2* derivatives [[buffer(1)]], const device uint* metadata [[buffer(2)]],
    const device uint* row_indices [[buffer(3)]], const device uint* offsets [[buffer(4)]],
    const device uint* active [[buffer(5)]], const device uint2* candidates [[buffer(6)]],
    device float4* statistics [[buffer(7)]], constant OrderedHistogramParams& p [[buffer(8)]],
    uint2 local [[thread_position_in_threadgroup]], uint2 group [[threadgroup_position_in_grid]]) {
    const uint tid = local.x, partition = active[group.x], candidate = group.y;
    const uint2 split = candidates[candidate];
    float4 lh = 0, ll = 0, rh = 0, rl = 0;
    for (uint position = offsets[partition] + tid; position < offsets[partition + 1]; position += 256) {
        const uint occurrence = row_indices[position], info = metadata[occurrence], row = info & 0x7fffffffu;
        const float2 d = derivatives[p.derivative_offset + occurrence];
        const float4 value = (info & 0x80000000u) ? float4(d.y, d.x, 0, 0) : float4(0, 0, d.y, d.x);
        const uint bin = bins[ulong(split.x) * p.rows + row], border = split.y & 0xffu;
        const bool right = (split.y & 0x80000000u) ? bin == border : bin > border;
        if (right) OrderedHistogramMerge(rh, rl, value);
        else OrderedHistogramMerge(lh, ll, value);
    }
    threadgroup float4 left_high[256], left_low[256], right_high[256], right_low[256];
    left_high[tid] = lh; left_low[tid] = ll; right_high[tid] = rh; right_low[tid] = rl;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = 128; stride; stride >>= 1) {
        if (tid < stride) {
            lh = left_high[tid]; ll = left_low[tid]; rh = right_high[tid]; rl = right_low[tid];
            OrderedHistogramMerge(lh, ll, left_high[tid + stride]); OrderedHistogramMerge(lh, ll, left_low[tid + stride]);
            OrderedHistogramMerge(rh, rl, right_high[tid + stride]); OrderedHistogramMerge(rh, rl, right_low[tid + stride]);
            left_high[tid] = lh; left_low[tid] = ll; right_high[tid] = rh; right_low[tid] = rl;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (!tid) {
        const uint leaf = partition / p.fold_slots, fold = partition % p.fold_slots, leaves = p.partitions / p.fold_slots;
        const ulong output = ((ulong(candidate) * leaves + leaf) * p.fold_count + fold) * 2;
        statistics[output] = left_high[0] + left_low[0]; statistics[output + 1] = right_high[0] + right_low[0];
    }
}
)METAL";
