#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "../native/metal_kernels.h"
#include "../native/metal_additional_objective_kernels.h"
#include "../native/metal_objective_kernels.h"
#include "../native/metal_deep_partition_kernels.h"
#include "../native/metal_ordered_histogram_kernels.h"
#include "../native/metal_ordered_histogram_runtime.h"
#include <algorithm>
#include <cstring>
#include <initializer_list>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace {
struct ProbeParams {
    uint32_t Rows, Features, Folds, Leaves, Candidates, PackedRows, Reuse, CandidateBatch;
    uint64_t TileBudget;
    uint32_t HistogramMaxLeaves, Reserved;
};
static_assert(sizeof(ProbeParams) == 48 && CBMMetalKernelAbiVersion == 2);
void Require(bool condition, const std::string& text) {
    if (!condition) throw std::invalid_argument(text);
}
std::string Error(NSError* error) { return error ? error.localizedDescription.UTF8String : "unknown Metal error"; }
struct Runtime {
    id<MTLDevice> Device;
    id<MTLCommandQueue> Queue;
    std::unordered_map<std::string, id<MTLComputePipelineState>> Pipelines;
    Runtime() {
        Device = MTLCreateSystemDefaultDevice();
        Require(Device != nil, "No Metal device");
        Queue = [Device newCommandQueue];
        MTLCompileOptions* options = [MTLCompileOptions new];
        options.languageVersion = MTLLanguageVersion3_0;
        options.fastMathEnabled = NO;
        NSString* source = [NSString stringWithFormat:@"%s\n%s\n%s\n%s\n%s",
            CBMMetalSource, CBMMetalAdditionalObjectiveSource, CBMMetalObjectiveSource,
            CBMMetalDeepPartitionSource, CBMMetalOrderedHistogramSource];
        NSError* error = nil;
        id<MTLLibrary> library = [Device newLibraryWithSource:source options:options error:&error];
        Require(library != nil, "Ordered histogram shader compilation: " + Error(error));
        for (const char* name : {"InitializeOrderedHistogramOccurrences", "UpdateOrderedHistogramLeafIds",
                "ResetOrderedHistogramJobs", "BuildOrderedHistogramJobs", "OrderedHistogramArguments",
                "ClearOrderedHistogram", "ComputeOrderedHistogram", "ScanOrderedHistogram",
                "SubtractOrderedHistogramSibling", "ExtractOrderedHistogramCandidates",
                "CountDeepPartitionBits", "ScanDeepPartitionTiles", "ScanDeepPartitionBlocks",
                "BuildDeepPartitionOffsets", "ScatterDeepPartitionRows", "ClearOrderedPartitionCandidateStatistics",
                "ComputeOrderedPartitionCandidates"}) {
            id<MTLFunction> function = [library newFunctionWithName:[NSString stringWithUTF8String:name]];
            Require(function != nil, std::string("Missing histogram kernel ") + name);
            auto pipeline = [Device newComputePipelineStateWithFunction:function error:&error];
            Require(pipeline != nil, Error(error));
            Pipelines.emplace(name, pipeline);
        }
    }
    id<MTLBuffer> Buffer(uint64_t bytes, const void* data = nullptr) {
        bytes = std::max<uint64_t>(bytes, 4);
        id<MTLBuffer> result = data ? [Device newBufferWithBytes:data length:bytes options:MTLResourceStorageModeShared]
            : [Device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
        Require(result != nil, "Histogram probe buffer allocation failed");
        return result;
    }
};
Runtime& Context() { static Runtime context; return context; }
struct Binding {
    id<MTLBuffer> Buffer;
    uint64_t Offset;
    Binding(id<MTLBuffer> buffer, uint64_t offset = 0) : Buffer(buffer), Offset(offset) {}
};
struct Command {
    using BindingType = Binding;
    id<MTLCommandBuffer> Buffer;
    uint64_t& Dispatches;
    explicit Command(uint64_t& dispatches) : Buffer([Context().Queue commandBuffer]), Dispatches(dispatches) {}
    template<class P> void Dispatch(const char* name, std::initializer_list<Binding> inputs, const P& p,
        uint64_t width, bool grouped = false, uint64_t height = 1,
        id<MTLBuffer> arguments = nil, uint32_t argumentOffset = 0) {
        Require(width && height, "Empty histogram probe dispatch");
        auto encoder = [Buffer computeCommandEncoder];
        [encoder setComputePipelineState:Context().Pipelines.at(name)];
        uint32_t index = 0;
        for (auto input : inputs) [encoder setBuffer:input.Buffer offset:input.Offset atIndex:index++];
        [encoder setBytes:&p length:sizeof(p) atIndex:index];
        if (arguments) [encoder dispatchThreadgroupsWithIndirectBuffer:arguments indirectBufferOffset:argumentOffset
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        else if (grouped) [encoder dispatchThreadgroups:MTLSizeMake(width, height, 1)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        else [encoder dispatchThreads:MTLSizeMake(width, height, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [encoder endEncoding];
        ++Dispatches;
    }
    template<class P> void DispatchIndirect(const char* name, std::initializer_list<Binding> inputs,
        const P& p, id<MTLBuffer> arguments, uint32_t argumentOffset) {
        Dispatch(name, inputs, p, 1, true, 1, arguments, argumentOffset);
    }
    void Wait() {
        [Buffer commit]; [Buffer waitUntilCompleted];
        Require(Buffer.status == MTLCommandBufferStatusCompleted, "Histogram probe execution: " + Error(Buffer.error));
    }
};
}

extern "C" int cbm_ordered_histogram_probe(const ProbeParams* params, const uint8_t* bins,
    const uint32_t* permutation, const uint32_t* folds, const float* sampled,
    const uint32_t* originalLeafIds, const uint32_t* candidates,
    float* statistics, uint64_t* diagnostics, char* error, uint32_t errorCapacity) {
    @autoreleasepool {
        try {
            Require(params && bins && permutation && folds && sampled && originalLeafIds && candidates &&
                statistics && diagnostics, "Histogram probe buffers required");
            const ProbeParams& p = *params;
            Require(p.Rows >= 4 && p.Rows <= (1u << 24) && p.Features && p.Features <= 4096 &&
                p.Folds && p.Folds <= 4096 && p.Leaves && p.Leaves <= 65536 &&
                (p.Leaves & (p.Leaves - 1)) == 0 && p.Candidates && p.Reuse <= 1 && p.TileBudget,
                "Invalid histogram probe configuration");
            uint32_t foldSlots = 1;
            while (foldSlots < p.Folds) foldSlots *= 2;
            const uint64_t maxPartitions = uint64_t(foldSlots) * p.Leaves;
            Require(maxPartitions <= UINT32_MAX, "Histogram partition count overflow");
            const uint32_t derivativeOffset = folds[2];
            uint64_t end = derivativeOffset;
            for (uint32_t fold = 0; fold < p.Folds; ++fold) {
                Require(folds[fold * 4] && folds[fold * 4] <= folds[fold * 4 + 1] &&
                    folds[fold * 4 + 1] <= p.Rows && folds[fold * 4 + 2] == end,
                    "Invalid histogram fold descriptor");
                end += folds[fold * 4 + 1];
            }
            Require(end <= p.PackedRows && end > derivativeOffset, "Invalid histogram derivative capacity");
            const uint32_t occurrences = end - derivativeOffset;
            std::vector<bool> seen(p.Rows);
            std::vector<uint8_t> featureMajor(uint64_t(p.Rows) * p.Features);
            for (uint32_t row = 0; row < p.Rows; ++row) {
                Require(permutation[row] < p.Rows && !seen[permutation[row]] && originalLeafIds[row] < p.Leaves,
                    "Invalid histogram permutation or leaf ID");
                seen[permutation[row]] = true;
                for (uint32_t feature = 0; feature < p.Features; ++feature)
                    featureMajor[uint64_t(feature) * p.Rows + row] = bins[uint64_t(row) * p.Features + feature];
            }
            std::vector<uint32_t> features(p.Candidates), borders(p.Candidates);
            std::vector<uint8_t> types(p.Candidates);
            for (uint32_t c = 0; c < p.Candidates; ++c) {
                features[c] = candidates[c * 2]; borders[c] = candidates[c * 2 + 1] & 0xffu;
                types[c] = candidates[c * 2 + 1] >> 31;
                Require((candidates[c * 2 + 1] & 0x7fffff00u) == 0, "Invalid packed candidate type/border");
            }
            CBMOrderedHistogramPlan plan(p.Rows, occurrences, p.Folds, p.Leaves, p.Features,
                p.Candidates, features.data(), borders.data(), p.TileBudget, p.Reuse, p.HistogramMaxLeaves, types.data());
            const uint64_t statisticBytes = uint64_t(p.Candidates) * p.Leaves * p.Folds * 32;
            Require(plan.Bytes + statisticBytes + uint64_t(p.Rows) * p.Features < (1ull << 29),
                "Histogram probe workspace exceeds 512 MiB");
            auto& runtime = Context();
            auto gpuBins = runtime.Buffer(featureMajor.size(), featureMajor.data());
            auto gpuPermutation = runtime.Buffer(uint64_t(p.Rows) * 4, permutation);
            auto gpuFolds = runtime.Buffer(uint64_t(p.Folds) * 16, folds);
            auto gpuDerivatives = runtime.Buffer(uint64_t(p.PackedRows) * 8, sampled);
            auto gpuOriginalLeaves = runtime.Buffer(uint64_t(p.Rows) * 4, originalLeafIds);
            auto gpuCandidates = runtime.Buffer(uint64_t(p.Candidates) * 8, candidates);
            auto output = runtime.Buffer(statisticBytes);
            CBMOrderedHistogramWorkspace workspace(runtime, std::move(plan));
            std::fill(diagnostics, diagnostics + 5, 0);
            diagnostics[0] = workspace.Plan.Tiles.size();
            Command initialize(diagnostics[3]);
            workspace.Initialize(initialize, Binding(gpuPermutation), Binding(gpuFolds), derivativeOffset);
            initialize.Wait();
            for (uint32_t leaves = 1; leaves <= p.Leaves; leaves *= 2) {
                Command prepare(diagnostics[3]);
                if (leaves > 1) workspace.Partition(prepare, gpuOriginalLeaves, leaves);
                workspace.Prepare(prepare, leaves);
                prepare.Wait(); workspace.Check();
                const uint32_t* state = static_cast<const uint32_t*>(workspace.State.contents);
                diagnostics[1] += state[0];
                diagnostics[2] += workspace.P.Reuse ? state[1] : 0;
                if (!workspace.CanHistogram()) {
                    ++diagnostics[4];
                    if (leaves == p.Leaves) {
                        const uint64_t stride = uint64_t(leaves) * p.Folds * 8;
                        const uint32_t batch = p.CandidateBatch ? std::min(p.CandidateBatch, p.Candidates) : p.Candidates;
                        for (uint32_t begin = 0; begin < p.Candidates; begin += batch) {
                            const uint32_t count = std::min(batch, p.Candidates - begin);
                            Command direct(diagnostics[3]);
                            workspace.DirectCandidates(direct, gpuBins, gpuDerivatives,
                                Binding(gpuCandidates, uint64_t(begin) * 8), count, output);
                            direct.Wait(); workspace.Check();
                            std::memcpy(statistics + uint64_t(begin) * stride, output.contents, uint64_t(count) * stride * 4);
                        }
                    }
                    continue;
                }
                for (const auto& tile : workspace.Plan.Tiles) {
                    Command histogram(diagnostics[3]);
                    workspace.Compute(histogram, tile, gpuBins, gpuDerivatives);
                    histogram.Wait(); workspace.Check();
                    if (leaves == p.Leaves) {
                        const uint64_t stride = uint64_t(leaves) * p.Folds * 8;
                        const uint32_t batch = p.CandidateBatch ? std::min(p.CandidateBatch, tile.Candidates) : tile.Candidates;
                        for (uint32_t begin = 0; begin < tile.Candidates; begin += batch) {
                            const uint32_t count = std::min(batch, tile.Candidates - begin);
                            Command extract(diagnostics[3]);
                            workspace.Extract(extract, tile, begin, count, output);
                            extract.Wait();
                            const float* source = static_cast<const float*>(output.contents);
                            for (uint32_t local = 0; local < count; ++local) {
                                const uint32_t global = workspace.Plan.CandidateIndices[tile.FirstCandidate + begin + local];
                                std::memcpy(statistics + uint64_t(global) * stride, source + uint64_t(local) * stride, stride * 4);
                            }
                        }
                    }
                }
            }
            return 0;
        } catch (const std::exception& e) {
            if (error && errorCapacity) { std::strncpy(error, e.what(), errorCapacity - 1); error[errorCapacity - 1] = 0; }
            return 1;
        }
    }
}
