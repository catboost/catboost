// microbench_contention.cpp — Sprint 19 T3b contention sensitivity sweep
//
// T3b (no-shuffle + threadgroup atomic-CAS) depends on contention rate at
// each bin. At 128 bins (gate config), avg docs/bin = 50000/128 ≈ 390.
// At 32 bins, avg docs/bin = 1562 — 4× higher contention. If T3b's
// atomic-CAS cost degrades with contention, it may not hold up at narrower
// bin counts or at deeper tree depths (fewer docs per partition).
//
// This harness measures T3b across {128, 64, 32} bin configs and compares
// against T0 production at the same bin count.
//
// COMPILE
//   clang++ -std=c++17 -O2 \
//     -I/opt/homebrew/Cellar/mlx/0.31.1/include \
//     -L/opt/homebrew/Cellar/mlx/0.31.1/lib -lmlx \
//     -framework Metal -framework Foundation \
//     docs/sprint19/scratch/algorithmic/microbench_contention.cpp \
//     -o /tmp/microbench_contention

#include <mlx/mlx.h>
#include <mlx/fast.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <numeric>
#include <string>
#include <vector>

namespace mx = mlx::core;

static constexpr uint32_t N_DOCS    = 50000;
static constexpr int      WARM_RUNS = 5;
static constexpr int      TIMED_RUNS= 5;

struct TTimingResult { double mean_ms=0, stdev_ms=0; std::vector<double> runs_ms; };
static TTimingResult Stats(const std::vector<double>& ms) {
    TTimingResult r; r.runs_ms = ms;
    double s=0; for (double v : ms) s+=v; r.mean_ms = s/ms.size();
    double var=0; for (double v : ms) var += (v-r.mean_ms)*(v-r.mean_ms);
    r.stdev_ms = ms.size()>1 ? std::sqrt(var/(ms.size()-1)) : 0;
    return r;
}
template<typename F>
static TTimingResult Time(const char* name, F&& fn, int w, int t) {
    fprintf(stderr, "  %s: ", name);
    for (int i=0;i<w;++i) fn();
    std::vector<double> ts;
    for (int i=0;i<t;++i) {
        auto t0 = std::chrono::steady_clock::now();
        fn();
        auto t1 = std::chrono::steady_clock::now();
        ts.push_back(std::chrono::duration<double, std::milli>(t1-t0).count());
        fprintf(stderr, ".");
    }
    fprintf(stderr, "\n");
    return Stats(ts);
}

static const std::string kHeader = R"metal(
#include <metal_stdlib>
using namespace metal;
constant constexpr uint SIMD_SIZE         = 32;
constant constexpr uint FEATURES_PER_PACK = 4;
constant constexpr uint BINS_PER_BYTE     = 256;
constant constexpr uint BLOCK_SIZE        = 256;
constant constexpr uint NUM_SIMD_GROUPS   = 8;
constant constexpr uint HIST_PER_SIMD     = FEATURES_PER_PACK * BINS_PER_BYTE;
)metal";

static const std::string kT0 = R"metal(
    threadgroup float simdHist[NUM_SIMD_GROUPS][HIST_PER_SIMD];
    const uint tid=thread_index_in_threadgroup;
    const uint lane=tid & (SIMD_SIZE-1u);
    const uint simd_id=tid>>5u;
    for (uint b=lane; b<HIST_PER_SIMD; b+=SIMD_SIZE) simdHist[simd_id][b]=0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint numDocs=totalNumDocs;
    for (uint bs=simd_id*SIMD_SIZE; bs<numDocs; bs+=NUM_SIMD_GROUPS*SIMD_SIZE) {
        const uint d=bs+lane; const bool valid=(d<numDocs);
        uint packed=0u; float stat=0.0f;
        if (valid) {
            const uint docIdx=docIndices[d];
            packed=compressedIndex[docIdx];
            stat  =stats[docIdx];
        }
        for (uint src=0u; src<SIMD_SIZE; ++src) {
            const uint p_s=simd_shuffle(packed,src);
            const float s_s=simd_shuffle(stat,src);
            const bool v_s=simd_shuffle(valid,src);
            if (!v_s) continue;
            for (uint f=0u; f<FEATURES_PER_PACK; ++f) {
                const uint bin=(p_s>>(24u-8u*f))&0xFFu;
                if (bin<foldCounts[f]+1u && (bin&(SIMD_SIZE-1u))==lane)
                    simdHist[simd_id][f*BINS_PER_BYTE+bin]+=s_s;
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid<HIST_PER_SIMD) output[tid]=simdHist[0][tid];
)metal";

static const std::string kT3b = R"metal(
    threadgroup atomic_uint simdHistU[HIST_PER_SIMD];
    const uint tid=thread_index_in_threadgroup;
    for (uint b=tid; b<HIST_PER_SIMD; b+=BLOCK_SIZE)
        atomic_store_explicit(&simdHistU[b], 0u, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint numDocs=totalNumDocs;
    for (uint d=tid; d<numDocs; d+=BLOCK_SIZE) {
        const uint docIdx=docIndices[d];
        const uint packed=compressedIndex[docIdx];
        const float stat=stats[docIdx];
        for (uint f=0u; f<FEATURES_PER_PACK; ++f) {
            const uint bin=(packed>>(24u-8u*f))&0xFFu;
            if (bin<foldCounts[f]+1u) {
                const uint idx=f*BINS_PER_BYTE+bin;
                uint oldBits=atomic_load_explicit(&simdHistU[idx], memory_order_relaxed);
                for (;;) {
                    const float oldF=as_type<float>(oldBits);
                    const float newF=oldF+stat;
                    const uint newBits=as_type<uint>(newF);
                    uint exp=oldBits;
                    if (atomic_compare_exchange_weak_explicit(
                            &simdHistU[idx], &exp, newBits,
                            memory_order_relaxed, memory_order_relaxed))
                        break;
                    oldBits=exp;
                }
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid<HIST_PER_SIMD) {
        const uint bits=atomic_load_explicit(&simdHistU[tid], memory_order_relaxed);
        output[tid]=as_type<float>(bits);
    }
)metal";

struct TResult { uint32_t bins; TTimingResult t0; TTimingResult t3b; };

static TResult RunBinConfig(uint32_t numBins) {
    // Host data — bins mod numBins so all bins get density
    std::vector<uint32_t> ci(N_DOCS, 0u);
    for (uint32_t d = 0; d < N_DOCS; ++d) {
        const uint32_t b0 = (d * 7u)  % numBins;
        const uint32_t b1 = (d * 11u) % numBins;
        const uint32_t b2 = (d * 13u) % numBins;
        const uint32_t b3 = (d * 17u) % numBins;
        ci[d] = (b0<<24)|(b1<<16)|(b2<<8)|b3;
    }
    std::vector<uint32_t> di(N_DOCS); std::iota(di.begin(), di.end(), 0u);
    std::vector<float> st(N_DOCS);
    for (uint32_t d=0; d<N_DOCS; ++d) st[d] = float(d%128)/128.0f;
    std::vector<uint32_t> fc(4, numBins);

    auto ciArr = mx::array(reinterpret_cast<const int32_t*>(ci.data()),
                           {static_cast<int>(N_DOCS)}, mx::uint32);
    auto diArr = mx::array(reinterpret_cast<const int32_t*>(di.data()),
                           {static_cast<int>(N_DOCS)}, mx::uint32);
    auto stArr = mx::array(st.data(), {static_cast<int>(N_DOCS)}, mx::float32);
    auto fcArr = mx::array(reinterpret_cast<const int32_t*>(fc.data()),
                           {4}, mx::uint32);
    auto tdArr = mx::array(static_cast<uint32_t>(N_DOCS), mx::uint32);
    mx::eval({ciArr, diArr, stArr, fcArr, tdArr});
    auto probeOut = mx::zeros({1024}, mx::float32);
    auto grid = std::make_tuple(256,1,1);
    auto tg = std::make_tuple(256,1,1);

    auto t0K = mx::fast::metal_kernel(
        (std::string("t0_")+std::to_string(numBins)).c_str(),
        {"compressedIndex","stats","docIndices","foldCounts","totalNumDocs"},
        {"output"}, kT0, kHeader, true, false);
    auto t3bK = mx::fast::metal_kernel(
        (std::string("t3b_")+std::to_string(numBins)).c_str(),
        {"compressedIndex","stats","docIndices","foldCounts","totalNumDocs"},
        {"output"}, kT3b, kHeader, true, false);

    auto run = [&](auto& k, const std::vector<mx::array>& inputs) {
        auto r = k(inputs, {probeOut.shape()}, {mx::float32}, grid, tg, {},
                   0.0f, false, mx::Device::gpu);
        mx::eval(r[0]);
    };

    TResult res{numBins};
    res.t0  = Time((std::string("T0_")+std::to_string(numBins)).c_str(),
        [&](){ run(t0K, {ciArr, stArr, diArr, fcArr, tdArr}); },
        WARM_RUNS, TIMED_RUNS);
    res.t3b = Time((std::string("T3b_")+std::to_string(numBins)).c_str(),
        [&](){ run(t3bK, {ciArr, stArr, diArr, fcArr, tdArr}); },
        WARM_RUNS, TIMED_RUNS);
    return res;
}

int main() {
    fprintf(stderr, "[S19-contention] Sweeping bin count {128, 64, 32, 16}...\n");
    auto r128 = RunBinConfig(128);
    auto r64  = RunBinConfig(64);
    auto r32  = RunBinConfig(32);
    auto r16  = RunBinConfig(16);

    fprintf(stdout, "\nT3b contention sensitivity vs T0 (all 1 TG × 256 threads, N=50k)\n");
    fprintf(stdout, "%-6s  %10s  %10s  %10s  %10s\n",
            "bins", "T0_ms", "T3b_ms", "T3b/T0", "docs/bin");
    auto row = [&](const TResult& r) {
        const double ratio = r.t0.mean_ms>0 ? r.t3b.mean_ms/r.t0.mean_ms : 0;
        const double docsPerBin = double(N_DOCS)/double(r.bins);
        fprintf(stdout, "%-6u  %10.3f  %10.3f  %10.3f  %10.1f\n",
                r.bins, r.t0.mean_ms, r.t3b.mean_ms, ratio, docsPerBin);
    };
    row(r128);
    row(r64);
    row(r32);
    row(r16);
    return 0;
}
