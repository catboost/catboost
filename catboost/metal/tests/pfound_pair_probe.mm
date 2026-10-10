#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "../native/metal_pfound_pair_kernels.h"
#include "../native/metal_bootstrap_kernels.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <initializer_list>
#include <stdexcept>
#include <vector>

struct Params {
    uint32_t Rows,Groups,Tasks,Pairs,Permutations,SeedLow,SeedHigh,Reserved;
    float Decay,Temperature; uint32_t BootstrapType,AbsoluteIteration;
};
struct BootstrapParams {
    uint32_t Rows,Type,SeedLow,SeedHigh,Iteration,Stream,R0,R1;
    float Temperature,Subsample,Lambda,Noise;
};
static_assert(sizeof(Params)==48 && sizeof(BootstrapParams)==48);

extern "C" int cbm_pfound_pair_probe(const Params* input,const float* targets,const float* weights,
    const float* point,const uint32_t* offsets,const uint32_t* documentIds,
    float* exponents,float* rawMatrix,uint32_t* pairs,float* edges,float* gradients,
    char* error,uint32_t capacity) {
    @autoreleasepool {
        try {
            auto require=[](bool value,const char* message){if(!value)throw std::runtime_error(message);};
            require(input && targets && weights && point && offsets && documentIds && exponents && rawMatrix && pairs && edges && gradients,"Null PFound pair probe input/output");
            Params p=*input;
            require(p.Rows && p.Rows<=(1u<<24) && p.Groups && p.Groups<=p.Rows && p.Pairs<=(1u<<24)
                && p.Permutations && p.Permutations<=10000 && !p.Reserved && std::isfinite(p.Decay) && p.Decay>=0 && p.Decay<=1
                && std::isfinite(p.Temperature) && p.Temperature>=0 && p.BootstrapType<=1,
                "Invalid PFound pair dimensions, permutations, decay or bootstrap");
            require(offsets[0]==0 && offsets[p.Groups]==p.Rows,"PFound offsets must span rows");
            std::vector<uint32_t> qids(p.Rows),pairOffsets(p.Groups+1),tasks;
            uint64_t pairCount=0;
            for(uint32_t q=0;q<p.Groups;++q) {
                require(offsets[q]<offsets[q+1] && offsets[q+1]<=p.Rows && offsets[q+1]-offsets[q]<=1023,
                    "PFound sampled queries must contain 1..1023 rows");
                const uint64_t size=offsets[q+1]-offsets[q];pairCount+=size*(size-1)/2;
                require(pairCount<=p.Pairs,"PFound pair capacity is too small");pairOffsets[q+1]=pairCount;
                for(uint32_t row=offsets[q];row<offsets[q+1];++row) {
                    require(std::isfinite(targets[row]) && std::isfinite(weights[row]) && weights[row]>=0 && std::isfinite(point[row]),"Invalid PFound observation");
                    require(!row || documentIds[row]>documentIds[row-1],"PFound sampled document IDs must increase strictly");
                    qids[row]=q;
                }
            }
            require(pairCount==p.Pairs,"PFound pair capacity does not match query shapes");
            for(uint32_t q=0;q<p.Groups;) {
                const uint32_t limit=std::min(p.Rows,offsets[q]+1024),next=limit==p.Rows ? p.Groups : qids[limit];
                require(next>q,"Invalid PFound task packing");tasks.push_back(q);tasks.push_back(next);q=next;
            }
            p.Tasks=tasks.size()/2;
            const uint64_t bytes=32ull*p.Rows+32ull*std::max(p.Pairs,1u)+8ull*(p.Groups+1)+8ull*p.Tasks+4;
            require(bytes<=(1ull<<30),"PFound probe workspace exceeds 1 GiB");
            static id<MTLDevice> device=MTLCreateSystemDefaultDevice();require(device!=nil,"No Metal device");
            static id<MTLLibrary> library=[&] {
                MTLCompileOptions* options=[MTLCompileOptions new];options.languageVersion=MTLLanguageVersion3_0;options.fastMathEnabled=NO;
                NSError* message=nil;auto result=[device newLibraryWithSource:[NSString stringWithFormat:@"%s\n%s",CBMMetalPFoundPairSource,CBMMetalBootstrapSource] options:options error:&message];
                if(!result)throw std::runtime_error(message.localizedDescription.UTF8String);return result;
            }();
            auto buffer=[&](const void* source,uint64_t size) -> id<MTLBuffer> {
                size=std::max<uint64_t>(size,4);require(size<=device.maxBufferLength,"PFound buffer exceeds device limit");
                auto value=source ? [device newBufferWithBytes:source length:size options:MTLResourceStorageModeShared]
                    :[device newBufferWithLength:size options:MTLResourceStorageModeShared];
                require(value!=nil,"PFound allocation failed");if(!source)std::memset(value.contents,0,size);return value;
            };
            auto y=buffer(targets,4ull*p.Rows),w=buffer(weights,4ull*p.Rows),x=buffer(point,4ull*p.Rows),ids=buffer(documentIds,4ull*p.Rows);
            auto q=buffer(qids.data(),4ull*p.Rows),off=buffer(offsets,4ull*(p.Groups+1)),poff=buffer(pairOffsets.data(),4ull*(p.Groups+1)),task=buffer(tasks.data(),4ull*tasks.size());
            auto approx=buffer(nullptr,4ull*p.Rows),matrix=buffer(nullptr,4ull*p.Pairs),mult=buffer(nullptr,4ull*p.Pairs),outPairs=buffer(nullptr,8ull*p.Pairs);
            auto outEdges=buffer(nullptr,16ull*p.Pairs),outGradient=buffer(nullptr,8ull*p.Rows),status=buffer(nullptr,4);
            auto queue=[device newCommandQueue];auto command=[queue commandBuffer];require(queue && command,"PFound command allocation failed");
            auto dispatch=[&](NSString* name,std::initializer_list<id<MTLBuffer>> buffers,const void* params,uint32_t size,uint32_t count,bool grouped) {
                if(!count)return;NSError* message=nil;
                auto pipeline=[device newComputePipelineStateWithFunction:[library newFunctionWithName:name] error:&message];
                if(!pipeline)throw std::runtime_error(message.localizedDescription.UTF8String);
                require(pipeline.maxTotalThreadsPerThreadgroup>=256,"PFound requires 256 threads");
                auto encoder=[command computeCommandEncoder];[encoder setComputePipelineState:pipeline];NSUInteger index=0;
                for(auto value:buffers)[encoder setBuffer:value offset:0 atIndex:index++];[encoder setBytes:params length:size atIndex:index];
                [encoder dispatchThreadgroups:MTLSizeMake(grouped ? count : (count+255)/256,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];[encoder endEncoding];
            };
            BootstrapParams bootstrap={p.Pairs,p.BootstrapType,p.SeedLow,p.SeedHigh,p.AbsoluteIteration,0,0,0,p.Temperature,1,0,0};
            dispatch(@"PreparePFoundPairApprox",{x,off,approx},&p,sizeof(p),p.Groups,true);
            dispatch(@"GeneratePFoundPairWeights",{approx,y,q,off,task,poff,matrix},&p,sizeof(p),p.Tasks,true);
            dispatch(@"GenerateBootstrapWeights",{mult,w},&bootstrap,sizeof(bootstrap),p.Pairs,false);
            dispatch(@"FinalizePFoundPairs",{matrix,mult,y,w,approx,off,poff,ids,outPairs,outEdges,status},&p,sizeof(p),p.Pairs,false);
            dispatch(@"ReducePFoundPairGradients",{outEdges,q,off,poff,outGradient},&p,sizeof(p),p.Rows,false);
            [command commit];[command waitUntilCompleted];
            if(command.status!=MTLCommandBufferStatusCompleted)throw std::runtime_error(command.error.localizedDescription.UTF8String);
            require(!*static_cast<const uint32_t*>(status.contents),"PFound pair target overflowed finite float32");
            std::memcpy(exponents,approx.contents,4ull*p.Rows);std::memcpy(rawMatrix,matrix.contents,4ull*p.Pairs);
            std::memcpy(pairs,outPairs.contents,8ull*p.Pairs);std::memcpy(edges,outEdges.contents,16ull*p.Pairs);
            std::memcpy(gradients,outGradient.contents,8ull*p.Rows);return 0;
        } catch(const std::exception& e) {
            if(error && capacity){std::strncpy(error,e.what(),capacity-1);error[capacity-1]=0;}return 1;
        }
    }
}
