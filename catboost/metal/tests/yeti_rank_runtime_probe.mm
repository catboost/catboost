#include "../native/metal_yeti_rank_runtime.h"
#include <memory>

namespace {
struct Probe {
    id<MTLDevice> Device = MTLCreateSystemDefaultDevice();
    id<MTLCommandQueue> Queue = [Device newCommandQueue];
    std::unique_ptr<CBMYetiRankRuntime> Target;
    uint32_t Rows;
    id<MTLBuffer> Cursor, Leaves, Ids, Labels, Weights, Gradient, Mass;
    Probe(uint32_t rows, uint32_t groups, const uint32_t* offsets,
        uint32_t permutations, float decay, bool legacy) : Rows(rows) {
        Target = std::make_unique<CBMYetiRankRuntime>(Device, rows, groups, offsets, permutations, decay, legacy);
        auto buffer = [&](uint64_t bytes) {
            auto result = [Device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
            if (!result) throw std::runtime_error("Probe buffer allocation failed");
            return result;
        };
        Cursor=buffer(4ull*rows); Ids=buffer(4ull*rows); Labels=buffer(4ull*rows);
        Weights=buffer(4ull*rows); Gradient=buffer(4ull*rows); Mass=buffer(4ull*rows);
        Leaves=buffer(4ull*65536);
    }
};
void Error(char* text, uint32_t capacity, const std::exception& error) {
    if (text && capacity) { std::strncpy(text,error.what(),capacity-1); text[capacity-1]=0; }
}
}

extern "C" void* cbm_yeti_runtime_create(uint32_t rows, uint32_t groups,
    const uint32_t* offsets, uint32_t permutations, float decay, uint32_t legacy,
    uint64_t* bytes, char* text, uint32_t capacity) {
    @autoreleasepool {
        try {
            auto probe=std::make_unique<Probe>(rows,groups,offsets,permutations,decay,legacy);
            *bytes=probe->Target->AllocatedBytes();
            return probe.release();
        } catch (const std::exception& error) { Error(text,capacity,error); return nullptr; }
    }
}
extern "C" void cbm_yeti_runtime_destroy(void* handle) { delete static_cast<Probe*>(handle); }
extern "C" int cbm_yeti_runtime_center(void* handle, float* leaves, uint32_t count,
    char* text, uint32_t capacity) {
    @autoreleasepool {
        try {
            auto& p=*static_cast<Probe*>(handle);
            if (!count || count>65536) throw std::runtime_error("Invalid probe leaf count");
            std::memcpy(p.Leaves.contents,leaves,4ull*count);
            auto command=[p.Queue commandBuffer];
            p.Target->EncodeCenterLeafValues(command,p.Leaves,count);
            [command commit]; [command waitUntilCompleted];
            if (command.status!=MTLCommandBufferStatusCompleted)
                throw std::runtime_error([[command.error localizedDescription] UTF8String]);
            p.Target->CheckStatus();
            std::memcpy(leaves,p.Leaves.contents,4ull*count);
            return 0;
        } catch (const std::exception& error) { Error(text,capacity,error); return 1; }
    }
}
extern "C" int cbm_yeti_runtime_step(void* handle, const float* cursor,
    const float* leaves, uint32_t leafCount, const uint32_t* ids, const float* labels,
    const float* weights, uint32_t applyShift, uint64_t seed, uint32_t mode,
    float* gradients, float* mass, uint64_t* dispatches, char* text, uint32_t capacity) {
    @autoreleasepool {
        try {
            auto& p=*static_cast<Probe*>(handle);
            if (!leafCount || leafCount>65536) throw std::runtime_error("Invalid probe leaf count");
            for (auto pair : {std::make_pair(p.Cursor,cursor),std::make_pair(p.Labels,labels),std::make_pair(p.Weights,weights)})
                std::memcpy(pair.first.contents,pair.second,4ull*p.Rows);
            std::memcpy(p.Leaves.contents,leaves,4ull*leafCount); std::memcpy(p.Ids.contents,ids,4ull*p.Rows);
            if (mode==1) p.Target->ClearStatus();
            auto command=[p.Queue commandBuffer];
            *dispatches=0;
            // Mode2 checks output alias rejection before any encoder is created.
            p.Target->EncodePointDerivatives(command,p.Cursor,p.Leaves,p.Ids,leafCount,applyShift,
                p.Labels,p.Weights,p.Gradient,mode==2?p.Gradient:p.Mass,seed,dispatches);
            [command commit]; [command waitUntilCompleted];
            if (command.status!=MTLCommandBufferStatusCompleted)
                throw std::runtime_error([[command.error localizedDescription] UTF8String]);
            p.Target->CheckStatus();
            std::memcpy(gradients,p.Gradient.contents,4ull*p.Rows); std::memcpy(mass,p.Mass.contents,4ull*p.Rows);
            return 0;
        } catch (const std::exception& error) { Error(text,capacity,error); return 1; }
    }
}
