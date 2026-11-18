#include "../native/metal_pairwise_matrix_runtime.h"

struct ProbeParams {
    uint32_t Rows,Pairs,Features,Candidates,Parents,Leaves,Iterations,Tile;
    uint32_t LeafMethod,ScoreMethod,Bootstrap,SeedLow,SeedHigh,AbsoluteIteration,R0,R1;
    float L2,NonDiag,Temperature,Subsample;
};
static_assert(sizeof(ProbeParams)==80);

extern "C" int cbm_pair_matrix_runtime_probe(const ProbeParams* parameters,uint64_t budget,
    const uint8_t* bins,const uint32_t* parentIds,const uint32_t* leafIds,const float* point,
    const uint32_t* winners,const uint32_t* losers,const float* pairWeights,const float* objectWeights,
    const uint32_t* sortedRows,const uint32_t* offsets,const uint32_t* features,const uint32_t* borders,const uint8_t* types,
    float* scores,float* leaves,float* weights,double* loss,uint64_t* allocated,uint64_t* dispatches,uint32_t* tile,
    char* errorText,uint32_t capacity) {
    @autoreleasepool {
        try {
            if (!parameters) throw std::runtime_error("Missing runtime parameters");
            const auto& p=*parameters;
            if (!p.Iterations || p.Iterations>100 || p.LeafMethod>1 || p.ScoreMethod>1 || p.R0 || p.R1)
                throw std::runtime_error("Invalid runtime leaf iterations or method");
            id<MTLDevice> device=MTLCreateSystemDefaultDevice();
            CBMPairwiseMatrixRuntime runtime(device,p.Rows,p.Pairs,winners,losers,pairWeights,p.Leaves,p.Candidates,p.Tile,budget);
            auto buffer=[&](const void* source,uint64_t bytes)->id<MTLBuffer> {
                auto value=source ? [device newBufferWithBytes:source length:bytes options:MTLResourceStorageModeShared]
                    : [device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
                if (!value)throw std::runtime_error("Runtime probe allocation failed");
                if (!source)std::memset(value.contents,0,bytes);return value;
            };
            auto data=buffer(bins,uint64_t(p.Rows)*p.Features),parent=buffer(parentIds,4ull*p.Rows),ids=buffer(leafIds,4ull*p.Rows);
            auto cursor=buffer(point,4ull*p.Rows),object=buffer(objectWeights,4ull*p.Rows);
            auto rows=buffer(sortedRows,4ull*p.Rows),off=buffer(offsets,4ull*(p.Leaves+1));
            auto cf=buffer(features,4ull*p.Candidates),cb=buffer(borders,4ull*p.Candidates),ct=buffer(types,p.Candidates);
            auto values=buffer(nullptr,4ull*p.Leaves),leafWeight=buffer(nullptr,4ull*p.Leaves);
            id<MTLCommandQueue> queue=[device newCommandQueue];
            id<MTLCommandBuffer> command=[queue commandBuffer];
            CBMBootstrapOptions bootstrap={};bootstrap.bootstrap_type=p.Bootstrap;
            bootstrap.random_seed_low=p.SeedLow;bootstrap.random_seed_high=p.SeedHigh;
            bootstrap.bagging_temperature=p.Temperature;bootstrap.subsample=p.Subsample;
            *dispatches=0;runtime.ClearStatus();
            runtime.EncodeEdges(command,cursor,values,parent,p.Parents,false,&bootstrap,p.AbsoluteIteration,dispatches);
            runtime.EncodeCandidates(command,data,parent,cf,cb,ct,p.Features,p.Parents,p.ScoreMethod,p.L2,p.NonDiag,dispatches);
            runtime.EncodeLeafLayout(command,ids,p.Leaves,dispatches);
            runtime.EncodeLeafWeights(command,object,rows,off,leafWeight,p.Leaves,dispatches);
            for (uint32_t iteration=0;iteration<p.Iterations;++iteration) {
                runtime.EncodeEdges(command,cursor,values,ids,p.Leaves,true,nullptr,0,dispatches);
                runtime.EncodeLeafProjection(command,p.Leaves,p.LeafMethod,dispatches);
                runtime.EncodeLeafDirection(command,p.Leaves,p.L2,p.NonDiag,dispatches);
                runtime.EncodeLeafUpdate(command,values,leafWeight,values,p.Leaves,1,dispatches);
            }
            runtime.EncodeCenterSolvedPoint(command,values,p.Leaves,dispatches);
            runtime.EncodeEdges(command,cursor,values,ids,p.Leaves,true,nullptr,0,dispatches);
            runtime.EncodeLoss(command,ids,p.Leaves,false,dispatches);
            [command commit];[command waitUntilCompleted];
            if (command.status!=MTLCommandBufferStatusCompleted)throw std::runtime_error(command.error.localizedDescription.UTF8String);
            runtime.CheckStatus();const auto metric=runtime.ReadLoss();loss[0]=metric.first;loss[1]=metric.second;
            std::memcpy(scores,runtime.CandidateScores().contents,8ull*p.Candidates);
            std::memcpy(leaves,values.contents,4ull*p.Leaves);std::memcpy(weights,leafWeight.contents,4ull*p.Leaves);
            *allocated=runtime.AllocatedBytes();*tile=runtime.CandidateTile();
            return 0;
        } catch(const std::exception& e) {
            if(errorText&&capacity){std::strncpy(errorText,e.what(),capacity-1);errorText[capacity-1]='\0';}return 1;
        }
    }
}
