#pragma once

// Append after QueryCrossEntropy statistics and the compensated leaf solver.
// Complete point-diagonal + query-Laplacian candidate matrices, tiled across
// both candidates and queries. Accumulators persist across query tiles.
static const char* CBMMetalQCECandidateSource = R"METAL(
struct QCECandidateParams {
    uint rows, groups, parents, candidates, features, first_candidate, group_begin, group_count;
};
inline void QCECandidateAdd4(thread float4& high, thread float4& low, float4 value) {
    const float4 sum = high + value;
    const float4 virtual_value = sum - high;
    const float4 error = (high - (sum - virtual_value)) + (value - virtual_value) + low;
    high = sum + error;
    low = error - (high - sum);
}

// Cache layout [candidate][query-in-tile][child leaf][high/low float4].
// The complement is accumulated independently so one-leaf queries cancel
// exactly, including when curvature spans many orders of magnitude.
kernel void CacheQCECandidateLeafSums(const device float4* row_stats [[buffer(0)]],
    const device uint* offsets [[buffer(1)]], const device uchar* bins [[buffer(2)]],
    const device uint* parent_ids [[buffer(3)]], const device uint* features [[buffer(4)]],
    const device uint* borders [[buffer(5)]], const device uchar* types [[buffer(6)]],
    const device uchar* active_queries [[buffer(7)]], device float4* cache [[buffer(8)]],
    device atomic_uint* status [[buffer(9)]], constant QCECandidateParams& p [[buffer(10)]],
    uint3 job [[threadgroup_position_in_grid]], uint3 local [[thread_position_in_threadgroup]]) {
    const uint lane=local.x;
    threadgroup float4 high_parts[CBM_QCE_THREADS],low_parts[CBM_QCE_THREADS];
    const uint leaf=job.x,local_query=job.y,candidate=job.z,query=p.group_begin+local_query;
    // A zero feature count selects projection into an already fixed tree.
    const uint index=p.first_candidate+candidate;
    const uint feature=p.features ? features[index] : 0, border=p.features ? borders[index] : 0, type=p.features ? types[index] : 0;
    const uint leaves=p.features ? 2*p.parents : p.parents;
    const uint begin=offsets[query],end=offsets[query+1],row=begin+lane;
    float4 high=float4(0.0f),low=float4(0.0f);
    const bool metadata=(!p.features || feature<p.features) && border<=255 && type<=1 && begin<end && end<=p.rows && end-begin<=256;
    if (!metadata) { if (!lane)atomic_fetch_or_explicit(status+candidate,1u,memory_order_relaxed); }
    else if (row<end && active_queries[query]) {
        if (parent_ids[row]>=p.parents || !all(isfinite(row_stats[row])) || any(row_stats[row].yzw<0.0f))
            atomic_fetch_or_explicit(status+candidate,2u,memory_order_relaxed);
        else {
            const uint value=p.features ? bins[ulong(feature)*p.rows+row] : 0;
            // Match CUDA solver order; one-hot swaps the final split bit.
            const uint child=p.features ? 2*parent_ids[row]+uint(type ? value!=border : value>border) : parent_ids[row];
            const float4 s=row_stats[row];high=child==leaf ? float4(s.xyz,0.0f) : float4(0,0,0,s.z);
        }
    }
    high_parts[lane]=high;low_parts[lane]=low;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint width=CBM_QCE_THREADS/2;width;width>>=1) {
        if (lane<width) {
            high=high_parts[lane];low=low_parts[lane];
            QCECandidateAdd4(high,low,high_parts[lane+width]);QCECandidateAdd4(high,low,low_parts[lane+width]);
            high_parts[lane]=high;low_parts[lane]=low;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (!lane) {
        const ulong cell=(ulong(candidate)*p.group_count+local_query)*leaves+leaf;
        cache[2*cell]=high_parts[0];cache[2*cell+1]=low_parts[0];
    }
}

// One thread per matrix cell, reducing queries in their original order. Both
// additions are preserved in persistent accumulators, so query tile size does
// not change results. Caller clears accumulators once per candidate batch.
kernel void AccumulateQCECandidateMatrices(const device float4* cache [[buffer(0)]],
    const device float4* group_stats [[buffer(1)]], device float2* gradient [[buffer(2)]],
    device float2* hessian [[buffer(3)]], device atomic_uint* status [[buffer(4)]],
    constant QCECandidateParams& p [[buffer(5)]], uint index [[thread_position_in_grid]]) {
    const uint leaves=p.features ? 2*p.parents : p.parents,cells=leaves*leaves;
    if (index>=p.candidates*cells)return;
    const uint candidate=index/cells,a=index%cells/leaves,b=index%leaves;
    float2 h=hessian[index],g=a==b ? gradient[candidate*leaves+a] : float2(0.0f);
    for (uint query=0;query<p.group_count;++query) {
        const ulong start=(ulong(candidate)*p.group_count+query)*leaves;
        const float4 ah=cache[2*(start+a)],al=cache[2*(start+a)+1];
        const float4 bh=cache[2*(start+b)],bl=cache[2*(start+b)+1];
        const float total=group_stats[p.group_begin+query].y;
        float2 value=a==b ? float2(ah.y,al.y) : float2(0.0f);
        if (!isfinite(total) || total<0.0f)atomic_fetch_or_explicit(status+candidate,4u,memory_order_relaxed);
        if (total>1e-20f) {
            const float2 ac=float2(ah.z,al.z),other=a==b ? float2(ah.w,al.w) : -float2(bh.z,bl.z);
            value=LeafMatrixAdd(value,LeafMatrixDivide(LeafMatrixMultiply(ac,other),float2(total+1e-20f,0.0f)));
        }
        h=LeafMatrixAdd(h,value);
        if (a==b)g=LeafMatrixAdd(g,float2(ah.x,al.x));
    }
    if (!all(isfinite(h)) || !all(isfinite(g)))atomic_fetch_or_explicit(status+candidate,4u,memory_order_relaxed);
    hessian[index]=h;if (a==b)gradient[candidate*leaves+a]=g;
}

kernel void FinalizeQCECandidateMatrices(const device float2* sums [[buffer(0)]],
    const device float2* matrices [[buffer(1)]], device float* gradient [[buffer(2)]],
    device float* hessian [[buffer(3)]], constant QCECandidateParams& p [[buffer(4)]],
    uint index [[thread_position_in_grid]]) {
    const uint leaves=p.features ? 2*p.parents : p.parents;
    if (index<p.candidates*leaves*leaves)hessian[index]=matrices[index].x+matrices[index].y;
    if (index<p.candidates*leaves)gradient[index]=sums[index].x+sums[index].y;
}
)METAL";
