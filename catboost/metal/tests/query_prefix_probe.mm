#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "../native/metal_query_sampler_kernels.h"
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <vector>

extern "C" int cbm_query_prefix_probe(const uint32_t* input,uint32_t rows,uint32_t limit,uint32_t* output,char* error,uint32_t capacity) {
    @autoreleasepool {try {
        if(!input || !output || !rows || rows>(1u<<24) || limit==0xffffffffu)throw std::runtime_error("Invalid prefix bounds");
        auto device=MTLCreateSystemDefaultDevice();if(!device)throw std::runtime_error("No Metal device");
        static id<MTLLibrary> library=[&]{
            MTLCompileOptions* options=[MTLCompileOptions new];options.languageVersion=MTLLanguageVersion3_0;options.fastMathEnabled=NO;
            NSError* e=nil;auto result=[device newLibraryWithSource:[NSString stringWithUTF8String:CBMMetalQuerySamplerSource] options:options error:&e];
            if(!result)throw std::runtime_error(e.localizedDescription.UTF8String);return result;
        }();
        auto queue=[device newCommandQueue];auto command=[queue commandBuffer];
        if(!queue || !command)throw std::runtime_error("Prefix queue allocation failed");
        std::vector<id<MTLBuffer>> levels;std::vector<uint32_t> sizes={rows};
        levels.push_back([device newBufferWithBytes:input length:4ull*rows options:MTLResourceStorageModeShared]);
        if(!levels.back())throw std::runtime_error("Prefix input allocation failed");
        for(uint32_t i=0;i<rows;++i)if(input[i]>limit+1)throw std::runtime_error("Prefix input exceeds saturated bound");
        auto dispatch=[&](const char* name,id<MTLBuffer> values,id<MTLBuffer> parent,uint32_t count){
            NSError* e=nil;auto fn=[library newFunctionWithName:[NSString stringWithUTF8String:name]];
            auto pipeline=[device newComputePipelineStateWithFunction:fn error:&e];
            if(!pipeline)throw std::runtime_error(e.localizedDescription.UTF8String);
            uint32_t p[8]={count,0,0,limit,0,0,0,0};
            auto encoder=[command computeCommandEncoder];[encoder setComputePipelineState:pipeline];
            [encoder setBuffer:values offset:0 atIndex:0];[encoder setBuffer:parent offset:0 atIndex:1];[encoder setBytes:p length:32 atIndex:2];
            [encoder dispatchThreadgroups:MTLSizeMake((count+255)/256,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];[encoder endEncoding];
        };
        for(uint32_t n=rows;n>1;n=(n+255)/256){
            auto next=[device newBufferWithLength:4ull*((n+255)/256) options:MTLResourceStorageModeShared];
            if(!next)throw std::runtime_error("Prefix total allocation failed");
            dispatch("ScanQuerySamplerPrefix",levels.back(),next,n);levels.push_back(next);sizes.push_back((n+255)/256);
        }
        for(size_t level=levels.size();level>2;--level){const uint32_t i=level-3;dispatch("AddQuerySamplerPrefix",levels[i],levels[i+1],sizes[i]);}
        [command commit];[command waitUntilCompleted];
        if(command.status!=MTLCommandBufferStatusCompleted)throw std::runtime_error(command.error.localizedDescription.UTF8String);
        std::memcpy(output,levels[0].contents,4ull*rows);return 0;
    }catch(const std::exception& e){if(error && capacity){std::strncpy(error,e.what(),capacity-1);error[capacity-1]=0;}return 1;}}
}
