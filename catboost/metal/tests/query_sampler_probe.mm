#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "../native/metal_query_sampler_runtime.h"
#include <cstring>

struct ProbeParams { uint32_t Rows,Groups,Passes,MaxQuery,PairLimit,AllowFailures,Reserved0,Reserved1; };
extern "C" int cbm_query_sampler_probe(const ProbeParams* input,uint64_t budget,
    const uint32_t* offsets,const uint32_t* keys,const float* masks,const float* fractions,
    uint32_t* shapes,uint32_t* documents,uint32_t* qids,uint32_t* sampledOffsets,uint32_t* pairOffsets,
    uint32_t* rowMasks,uint64_t* bytes,char* error,uint32_t capacity) {
    @autoreleasepool {
        try {
            auto require=[](bool value,const char* message){if(!value)throw std::runtime_error(message);};
            require(input && offsets && keys && masks && fractions && shapes && documents && qids && sampledOffsets && pairOffsets && rowMasks && bytes,
                "Null query sampler probe buffer");
            const auto p=*input;require(p.Passes && p.Passes<=16 && p.AllowFailures<=1 && !p.Reserved0 && !p.Reserved1,"Invalid sampler probe repetition configuration");
            auto device=MTLCreateSystemDefaultDevice();CBMQuerySamplerRuntime runtime(device,p.Rows,p.Groups,offsets,budget);
            *bytes=runtime.AllocatedBytes();
            auto keyBuffer=[device newBufferWithLength:4ull*p.Rows options:MTLResourceStorageModeShared];
            auto maskBuffer=[device newBufferWithLength:4ull*p.Groups options:MTLResourceStorageModeShared];
            auto queue=[device newCommandQueue];require(keyBuffer && maskBuffer && queue,"Sampler probe input allocation failed");
            std::memset(shapes,0,16ull*p.Passes);std::memset(documents,0,4ull*p.Passes*p.Rows);std::memset(qids,0,4ull*p.Passes*p.Rows);
            std::memset(sampledOffsets,0,4ull*p.Passes*(p.Groups+1));std::memset(pairOffsets,0,4ull*p.Passes*(p.Groups+1));
            std::memset(rowMasks,0,4ull*p.Passes*p.Rows);
            for(uint32_t pass=0;pass<p.Passes;++pass) {
                std::memcpy(keyBuffer.contents,keys+uint64_t(pass)*p.Rows,4ull*p.Rows);
                std::memcpy(maskBuffer.contents,masks+uint64_t(pass)*p.Groups,4ull*p.Groups);
                auto command=[queue commandBuffer];runtime.Encode(command,keyBuffer,maskBuffer,fractions[pass],p.MaxQuery,p.PairLimit);
                [command commit];[command waitUntilCompleted];
                require(command.status==MTLCommandBufferStatusCompleted,"Query sampler GPU command failed");
                CBMQuerySamplerRuntime::Shape shape;
                try { shape=runtime.ReadShape(); }
                catch(const std::exception&) {if(!p.AllowFailures)throw;shapes[4ull*pass+3]=1;continue;}
                std::memcpy(shapes+4ull*pass,&shape,16);
                std::memcpy(documents+uint64_t(pass)*p.Rows,runtime.SampledDocuments().contents,4ull*shape.Rows);
                std::memcpy(qids+uint64_t(pass)*p.Rows,runtime.SampledQueryIds().contents,4ull*shape.Rows);
                std::memcpy(sampledOffsets+uint64_t(pass)*(p.Groups+1),runtime.SampledOffsets().contents,4ull*(shape.Groups+1));
                std::memcpy(pairOffsets+uint64_t(pass)*(p.Groups+1),runtime.SampledPairOffsets().contents,4ull*(shape.Groups+1));
                std::memcpy(rowMasks+uint64_t(pass)*p.Rows,runtime.SampledMask().contents,4ull*p.Rows);
            }
            return 0;
        } catch(const std::exception& e) {
            if(error && capacity){std::strncpy(error,e.what(),capacity-1);error[capacity-1]=0;}return 1;
        }
    }
}
