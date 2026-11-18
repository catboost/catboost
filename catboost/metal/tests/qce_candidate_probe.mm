#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "../native/metal_query_cross_entropy_kernels.h"
#include "../native/metal_leaf_matrix_kernels.h"
#include "../native/metal_pairwise_score_kernels.h"
#include "../native/metal_qce_candidate_kernels.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <initializer_list>
#include <stdexcept>
#include <string>
#include <unordered_map>
struct CandidateParams { uint32_t Rows,Groups,Parents,Candidates,Features,First,GroupBegin,GroupCount; };
struct LeafParams { uint32_t Leaves,Diagonal,R0,R1;float L2,NonDiag,MinWeight,Step; };
extern "C" int cbm_qce_candidate_probe(const CandidateParams* input,uint32_t queryTile,float l2,float nonDiag,
    const float* rowStats,const float* groupStats,const uint32_t* offsets,const uint8_t* bins,const uint32_t* parentIds,
    const uint32_t* features,const uint32_t* borders,const uint8_t* types,const uint8_t* active,
    float* gradients,float* hessians,float* solutions,float* scores,uint32_t* statuses,char* error,uint32_t errorCapacity) {
    @autoreleasepool {
        try {
            auto require=[](bool value,const char* message){if(!value)throw std::runtime_error(message);};
            require(input && rowStats && groupStats && offsets && bins && parentIds && features && borders && types && active &&
                gradients && hessians && solutions && scores && statuses,"Missing QCE candidate probe input");
            auto p=*input;const uint32_t leaves=2*p.Parents;
            require(p.Rows && p.Groups && p.Groups<=p.Rows && p.Features && p.Candidates && p.Candidates<=64 &&
                p.Parents && p.Parents<=128 && queryTile && queryTile<=p.Groups && !p.First && !p.GroupBegin && !p.GroupCount,
                "Invalid QCE candidate dimensions");
            require(offsets[0]==0 && offsets[p.Groups]==p.Rows,"Invalid QCE query coverage");
            for(uint32_t q=0;q<p.Groups;++q)require(offsets[q]<offsets[q+1] && offsets[q+1]-offsets[q]<=256 && active[q]<=1,"Invalid QCE query size or activity");
            require(std::isfinite(l2)&&l2>=0&&std::isfinite(nonDiag)&&nonDiag>=0,"Invalid QCE candidate regularization");
            require(uint64_t(p.Candidates)*leaves*(uint64_t(queryTile)*32+uint64_t(leaves)*24+16)<(1ull<<29),"QCE probe workspace exceeds 512 MiB");
            auto device=MTLCreateSystemDefaultDevice();require(device!=nil,"Missing Metal device");
            MTLCompileOptions* options=[MTLCompileOptions new];options.languageVersion=MTLLanguageVersion3_0;options.fastMathEnabled=NO;
            NSError* metalError=nil;
            NSString* source=[NSString stringWithFormat:@"%s\n%s\n%s\n%s",CBMMetalQueryCrossEntropySource,
                CBMMetalLeafMatrixSource,CBMMetalPairwiseScoreSource,CBMMetalQCECandidateSource];
            auto library=[device newLibraryWithSource:source options:options error:&metalError];
            if(!library)throw std::runtime_error(metalError.localizedDescription.UTF8String);
            auto buffer=[&](const void* source,uint64_t size)->id<MTLBuffer>{
                auto value=source?[device newBufferWithBytes:source length:size options:MTLResourceStorageModeShared]:
                    [device newBufferWithLength:size options:MTLResourceStorageModeShared];
                require(value!=nil,"QCE buffer allocation failed");if(!source)std::memset(value.contents,0,size);return value;
            };
            const uint64_t cells=uint64_t(p.Candidates)*leaves*leaves,values=uint64_t(p.Candidates)*leaves;
            auto rs=buffer(rowStats,16ull*p.Rows),qs=buffer(groupStats,16ull*p.Groups),off=buffer(offsets,4ull*(p.Groups+1));
            auto data=buffer(bins,uint64_t(p.Rows)*p.Features),ids=buffer(parentIds,4ull*p.Rows),cf=buffer(features,4ull*p.Candidates);
            auto cb=buffer(borders,4ull*p.Candidates),ct=buffer(types,p.Candidates),flags=buffer(active,p.Groups);
            auto cache=buffer(nullptr,32ull*p.Candidates*queryTile*leaves),g=buffer(nullptr,4*values),h=buffer(nullptr,4*cells);
            auto ga=buffer(nullptr,8*values),ha=buffer(nullptr,8*cells),work=buffer(nullptr,8*cells),direction=buffer(nullptr,4*values);
            auto result=buffer(nullptr,8ull*p.Candidates),status=buffer(nullptr,4ull*p.Candidates);
            auto queue=[device newCommandQueue];auto command=[queue commandBuffer];require(command!=nil,"Missing QCE command buffer");
            std::unordered_map<std::string,id<MTLComputePipelineState>> pipelines;
            auto dispatch=[&](const char* name,std::initializer_list<id<MTLBuffer>> buffers,const void* params,size_t bytes,MTLSize grid,bool groups){
                if(!pipelines.count(name)) {
                    auto fn=[library newFunctionWithName:[NSString stringWithUTF8String:name]];
                    auto pipeline=[device newComputePipelineStateWithFunction:fn error:&metalError];
                    if(!pipeline)throw std::runtime_error(metalError.localizedDescription.UTF8String);pipelines.emplace(name,pipeline);
                }
                auto encoder=[command computeCommandEncoder];require(encoder!=nil,"Missing QCE encoder");
                [encoder setComputePipelineState:pipelines.at(name)];NSUInteger i=0;
                for(auto value:buffers)[encoder setBuffer:value offset:0 atIndex:i++];
                [encoder setBytes:params length:bytes atIndex:i];
                if(groups)[encoder dispatchThreadgroups:grid threadsPerThreadgroup:MTLSizeMake(256,1,1)];
                else[encoder dispatchThreads:grid threadsPerThreadgroup:MTLSizeMake(256,1,1)];[encoder endEncoding];
            };
            for(uint32_t first=0;first<p.Groups;first+=queryTile) {
                p.GroupBegin=first;p.GroupCount=std::min(queryTile,p.Groups-first);
                dispatch("CacheQCECandidateLeafSums",{rs,off,data,ids,cf,cb,ct,flags,cache,status},&p,sizeof(p),MTLSizeMake(leaves,p.GroupCount,p.Candidates),true);
                dispatch("AccumulateQCECandidateMatrices",{cache,qs,ga,ha,status},&p,sizeof(p),MTLSizeMake(cells,1,1),false);
            }
            dispatch("FinalizeQCECandidateMatrices",{ga,ha,g,h},&p,sizeof(p),MTLSizeMake(cells,1,1),false);
            const LeafParams leaf={leaves,1,0,0,l2,nonDiag,1e-20f,1};
            dispatch("RegularizePairwiseSplitMatrix",{h,work},&leaf,sizeof(leaf),MTLSizeMake(p.Candidates,1,1),true);
            dispatch("SolveLeafMatrix",{work,g,direction,status},&leaf,sizeof(leaf),MTLSizeMake(p.Candidates,1,1),true);
            dispatch("ScorePairwiseSplitSolution",{h,g,direction,result},&leaf,sizeof(leaf),MTLSizeMake(p.Candidates,1,1),true);
            [command commit];[command waitUntilCompleted];
            if(command.status!=MTLCommandBufferStatusCompleted)throw std::runtime_error(command.error.localizedDescription.UTF8String);
            std::memcpy(gradients,g.contents,4*values);std::memcpy(hessians,h.contents,4*cells);std::memcpy(solutions,direction.contents,4*values);
            std::memcpy(scores,result.contents,8ull*p.Candidates);std::memcpy(statuses,status.contents,4ull*p.Candidates);return 0;
        }catch(const std::exception& e){if(error&&errorCapacity){std::strncpy(error,e.what(),errorCapacity-1);error[errorCapacity-1]=0;}return 1;}
    }
}
