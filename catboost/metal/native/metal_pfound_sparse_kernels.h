#pragma once

// Sparse PFound contributions: append after the PFound and bootstrap sources.
// Inner ranks/seeds match GeneratePFoundPairWeights; only output storage differs.
static const char* CBMMetalPFoundSparseSource = R"METAL(
// Dense generation rejects 0 * infinity even for pairs never encountered by
// a permutation. Sparse storage must retain that failure behavior. Hosts run
// this allocation-free check only above temperature 20: the largest possible
// exponential is -log(1e-20), whose twentieth power remains finite in float32.
kernel void ValidatePFoundDenseBootstrap(device atomic_uint* status [[buffer(0)]],
    constant BootstrapParams& p [[buffer(1)]], uint pair [[thread_position_in_grid]]) {
    if(pair>=p.rows)return;
    ulong seed=BootstrapSeedForItem(pair,p);
    const float exponential=-log(BootstrapUniform(seed)+1e-20f);
    if(!isfinite(pow(exponential,p.temperature)))
        atomic_fetch_or_explicit(status,1u,memory_order_relaxed);
}

kernel void GeneratePFoundSparseContributions(const device float* exponents [[buffer(0)]],
    const device float* relevance [[buffer(1)]], const device uint* query_ids [[buffer(2)]],
    const device uint* offsets [[buffer(3)]], const device uint2* tasks [[buffer(4)]],
    const device uint* pair_offsets [[buffer(5)]], device uint* keys_out [[buffer(6)]], device float* contributions [[buffer(7)]],
    device uint* original_indices [[buffer(8)]],
    constant PFoundPairParams& p [[buffer(9)]], uint tid [[thread_position_in_threadgroup]],
    uint task [[threadgroup_position_in_grid]]) {
    const uint first_query = tasks[task].x, end_query = tasks[task].y;
    const uint start = offsets[first_query], count = offsets[end_query] - start;
    threadgroup float approx[1024], relev[1024], keys[1024];
    threadgroup uint qids[1024], indices[1024];
    for (uint k = 0; k < 4; ++k) {
        const uint index = tid + 256 * k;
        approx[index] = index < count ? exponents[start + index] : 1000.0f;
        relev[index] = index < count ? relevance[start + index] : 1000.0f;
        qids[index] = index < count ? query_ids[start + index] : end_query;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint cuda_seed = p.seed_low + p.seed_high;
    uint seed = 127u * first_query + 16807u * tid + cuda_seed * (1u + first_query);
    PFoundAdvance(seed); seed += cuda_seed;
    for (uint i = 0; i < 3; ++i) PFoundAdvance(seed);
    for (uint permutation = 0; permutation < p.permutations; ++permutation) {
        for (uint k = 0; k < 4; ++k) {
            const uint index = tid + 256 * k;
            const float uniform = float(PFoundAdvance(seed)) * 2.328306435996595e-10f;
            keys[index] = (index < count ? approx[index] : -1000.0f) * (uniform / (1.000001f - uniform));
            indices[index] = index;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint length = 2; length <= 1024; length <<= 1) {
            for (uint stride = length >> 1; stride; stride >>= 1) {
                uint output[4];
                for (uint k = 0; k < 4; ++k) {
                    const uint slot = tid + 256 * k, partner = slot ^ stride;
                    const uint a = indices[slot], b = indices[partner];
                    const bool take_first = ((slot & length) == 0) == ((slot & stride) == 0);
                    output[k] = PFoundBefore(a, b, qids, keys) == take_first ? a : b;
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
                for (uint k = 0; k < 4; ++k) indices[tid + 256 * k] = output[k];
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
        }
        for (uint k = 0; k < 4; ++k) {
            const uint slot = tid + 256 * k;
            const uint output = permutation * p.rows + start + slot;
            if (slot < count) {
                keys_out[output] = 0xffffffffu;
                contributions[output] = 0.0f;
                original_indices[output] = output;
            }
            if (slot < count && slot && qids[slot] == qids[slot - 1]) {
                const uint left = indices[slot - 1], right = indices[slot], query = qids[slot];
                const uint begin = offsets[query] - start, rank = slot - begin;
                const float weight = (0.15f * pow(p.decay, float(rank - 1))) * abs(relev[left] - relev[right]) / float(p.permutations);
                keys_out[output] = pair_offsets[query] + PFoundPairIndex(left - begin, right - begin);
                contributions[output] = weight;
            }
            threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
        }
    }
}


// Stable sorting by the original dense pair index preserves permutation order
// within each pair. Only its first sorted slot owns the raw accumulated weight.
// Addition is sequential float32, matching the dense kernel's per-permutation +=.
kernel void ReducePFoundSparseContributions(const device uint* keys [[buffer(0)]],
    const device uint* indices [[buffer(1)]], const device float* contributions [[buffer(2)]],
    device float* weights [[buffer(3)]], device atomic_uint* status [[buffer(4)]],
    constant PFoundPairParams& p [[buffer(5)]], uint slot [[thread_position_in_grid]]) {
    const uint slots=p.rows*p.permutations;
    if(slot>=slots)return;
    weights[slot]=0.0f;const uint key=keys[slot];
    if(key==0xffffffffu || (slot && keys[slot-1]==key))return;
    uint begin=slot,end=slots;
    while(begin<end){const uint mid=begin+(end-begin)/2;if(keys[mid]<=key)begin=mid+1;else end=mid;}
    float sum=0.0f;
    for(uint i=slot;i<begin;++i)sum+=contributions[indices[i]];
    weights[slot]=sum;
    if(!isfinite(sum) || sum<0 || key>=p.pairs)atomic_fetch_or_explicit(status,1u,memory_order_relaxed);
}

// Bayesian sampling stays indexed by the original dense pair ID, not by the
// sparse slot/rank. This preserves existing stream draws after compaction.
kernel void BootstrapPFoundSparsePairs(const device uint* keys [[buffer(0)]],
    device float* multipliers [[buffer(1)]], constant BootstrapParams& p [[buffer(2)]],
    uint slot [[thread_position_in_grid]]) {
    if(slot>=p.rows)return;
    float multiplier=1.0f;
    if(keys[slot]!=0xffffffffu && p.type==1 && p.temperature!=0.0f){
        ulong seed=BootstrapSeedForItem(keys[slot],p);
        const float exponential=-log(BootstrapUniform(seed)+1e-20f);
        multiplier=p.temperature==1.0f ? exponential : pow(exponential,p.temperature);
    }
    multipliers[slot]=multiplier;
}

kernel void FinalizePFoundSparsePairs(const device uint* keys [[buffer(0)]],
    const device float* matrix [[buffer(1)]], const device float* multipliers [[buffer(2)]],
    const device float* relevance [[buffer(3)]], const device float* weights [[buffer(4)]],
    const device float* exponents [[buffer(5)]], const device uint* offsets [[buffer(6)]],
    const device uint* pair_offsets [[buffer(7)]], const device uint* document_ids [[buffer(8)]],
    device uint2* pairs [[buffer(9)]], device float4* edges [[buffer(10)]], device atomic_uint* status [[buffer(11)]],
    constant PFoundPairParams& p [[buffer(12)]], uint slot [[thread_position_in_grid]]) {
    if(slot>=p.rows*p.permutations)return;
    pairs[slot]=uint2(0);edges[slot]=float4(0);
    const uint index=keys[slot];
    if(index==0xffffffffu)return;
    if(index>=p.pairs){atomic_fetch_or_explicit(status,1u,memory_order_relaxed);return;}
    uint lo=0,hi=p.groups;
    while(lo<hi){const uint mid=(lo+hi+1)/2;if(pair_offsets[mid]<=index)lo=mid;else hi=mid-1;}
    const uint local=index-pair_offsets[lo];
    const uint b=uint((1.0f+sqrt(8.0f*float(local)+1.0f))*0.5f),a=local-b*(b-1)/2;
    const uint left=offsets[lo]+a,right=offsets[lo]+b;
    pairs[slot]=uint2(document_ids[left],document_ids[right]);
    const float raw=matrix[slot]*multipliers[slot];
    const float mass=abs(raw)>1e-20f ? raw*weights[left] : 0.0f;
    const float ax=exponents[left]+1e-20f,ay=exponents[right]+1e-20f;
    const float gradient=mass*(relevance[left]>relevance[right] ? ay : -ax)/(ax+ay);
    edges[slot]=float4(gradient,mass,mass,0.0f);
    if(!isfinite(raw) || !isfinite(mass) || !isfinite(gradient) || mass<0.0f)
        atomic_fetch_or_explicit(status,1u,memory_order_relaxed);
}
)METAL";
