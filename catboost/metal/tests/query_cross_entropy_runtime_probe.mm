#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "../native/metal_query_cross_entropy_runtime.h"

struct Params {
    uint32_t Rows,Groups,Features,Candidates,Parents,Leaves,Iterations,Tile,QueryTile,Bootstrap,SeedLow,SeedHigh,Absolute,InvalidTrial,R0,R1;
    float Alpha,L2,NonDiag,Subsample;
};
extern "C" int cbm_qce_runtime_probe(const Params* input,uint64_t budget,
    const float* targets,const float* weights,const float* point,const uint32_t* offsets,const float* scales,
    const uint8_t* bins,const uint32_t* parents,const uint32_t* leafIds,const uint32_t* rows,const uint32_t* leafOffsets,
    const uint32_t* features,const uint32_t* borders,const uint8_t* types,
    float* rowStats,float* queryStats,uint8_t* mask,float* scores,float* gradient,float* hessian,float* values,float* leafWeights,
    double* loss,double* trialLoss,uint64_t* bytes,uint64_t* dispatches,uint32_t* tiles,char* error,uint32_t errorCapacity) {
    @autoreleasepool {
        try {
            if(!input)throw std::runtime_error("Missing QCE probe parameters");const auto p=*input;
            auto device=MTLCreateSystemDefaultDevice();
            CBMQueryCrossEntropyRuntime runtime(device,p.Rows,p.Groups,targets,weights,offsets,scales,p.Alpha,p.Leaves,p.Candidates,p.Tile,p.QueryTile,budget,p.R0);
            auto buffer=[&](const void* source,uint64_t size)->id<MTLBuffer>{
                auto value=source ? [device newBufferWithBytes:source length:size options:MTLResourceStorageModeShared] :
                    [device newBufferWithLength:size options:MTLResourceStorageModeShared];
                if(!value)throw std::runtime_error("Probe allocation failed");if(!source)std::memset(value.contents,0,size);return value;
            };
            auto cursor=buffer(point,4ull*p.Rows),ids=buffer(leafIds,4ull*p.Rows),pid=buffer(parents,4ull*p.Rows),data=buffer(bins,uint64_t(p.Rows)*p.Features);
            auto cf=buffer(features,4ull*p.Candidates),cb=buffer(borders,4ull*p.Candidates),ct=buffer(types,p.Candidates);
            auto raw=buffer(nullptr,4ull*p.Leaves),lw=buffer(nullptr,4ull*p.Leaves),original=buffer(weights,4ull*p.Rows);
            auto order=buffer(rows,4ull*p.Rows),leafOff=buffer(leafOffsets,4ull*(p.Leaves+1));
            auto queue=[device newCommandQueue];auto command=[queue commandBuffer];*dispatches=0;
            auto wait=[&]{[command commit];[command waitUntilCompleted];if(command.status!=MTLCommandBufferStatusCompleted)throw std::runtime_error(command.error.localizedDescription.UTF8String);runtime.CheckStatus();};
            CBMBootstrapOptions bootstrap={};bootstrap.bootstrap_type=p.Bootstrap;bootstrap.subsample=p.Subsample;
            bootstrap.random_seed_low=p.SeedLow;bootstrap.random_seed_high=p.SeedHigh;bootstrap.bagging_temperature=1;
            runtime.EncodeEdges(command,cursor,raw,pid,p.Leaves,false,&bootstrap,p.Absolute,dispatches);
            runtime.EncodeCandidates(command,data,pid,cf,cb,ct,p.Features,p.Parents,false,p.L2,p.NonDiag,dispatches);wait();
            std::memcpy(rowStats,runtime.Statistics().contents,16ull*p.Rows);std::memcpy(queryStats,runtime.QueryStatistics().contents,16ull*p.Groups);
            std::memcpy(mask,runtime.QueryMask().contents,p.Groups);std::memcpy(scores,runtime.CandidateScores().contents,8ull*p.Candidates);
            command=[queue commandBuffer];runtime.EncodeLeafLayout(command,ids,p.Leaves,dispatches);
            runtime.EncodeLeafWeights(command,original,order,leafOff,lw,p.Leaves,dispatches);
            for(uint32_t i=0;i<p.Iterations;++i) {
                runtime.EncodeEdges(command,cursor,raw,ids,p.Leaves,true,nullptr,0,dispatches);
                runtime.EncodeLeafProjection(command,p.Leaves,false,dispatches);
                runtime.EncodeLeafDirection(command,p.Leaves,p.L2,p.NonDiag,dispatches);
                runtime.EncodeLeafUpdate(command,raw,lw,raw,p.Leaves,1,dispatches);
            }
            runtime.EncodeEdges(command,cursor,raw,ids,p.Leaves,true,nullptr,0,dispatches);
            runtime.EncodeLoss(command,ids,p.Leaves,false,dispatches);wait();
            auto objective=runtime.ReadLoss();loss[0]=objective.first;loss[1]=objective.second;
            std::memcpy(values,raw.contents,4ull*p.Leaves);std::memcpy(leafWeights,lw.contents,4ull*p.Leaves);
            std::memcpy(gradient,runtime.LeafGradient().contents,4ull*p.Leaves);std::memcpy(hessian,runtime.LeafHessian().contents,4ull*p.Leaves*p.Leaves);
            *trialLoss=0;
            if(p.InvalidTrial) {
                auto trial=buffer(nullptr,4ull*p.Leaves);std::fill_n(static_cast<float*>(trial.contents),p.Leaves,std::numeric_limits<float>::max());
                command=[queue commandBuffer];runtime.EncodeBeginTrial(command);
                runtime.EncodeEdges(command,cursor,trial,ids,p.Leaves,true,nullptr,0,dispatches,true);
                runtime.EncodeLoss(command,ids,p.Leaves,false,dispatches);wait();*trialLoss=runtime.ReadTrialLoss();
                command=[queue commandBuffer];runtime.EncodeEdges(command,cursor,raw,ids,p.Leaves,true,nullptr,0,dispatches);
                runtime.EncodeLoss(command,ids,p.Leaves,false,dispatches);wait();
                const auto recovered=runtime.ReadLoss();
                if(recovered!=objective)throw std::runtime_error("Invalid trial changed the accepted QCE objective");
            }
            *bytes=runtime.AllocatedBytes();tiles[0]=runtime.CandidateTile();tiles[1]=runtime.QueryTileSize();return 0;
        }catch(const std::exception& e){if(error && errorCapacity){std::strncpy(error,e.what(),errorCapacity-1);error[errorCapacity-1]=0;}return 1;}
    }
}
