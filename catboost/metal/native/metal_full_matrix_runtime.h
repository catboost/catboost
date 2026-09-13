#pragma once
#import <Metal/Metal.h>
#include "metal_trainer.h"
#include <cstdint>
#include <utility>

// Common resident contract for non-diagonal CUDA targets. Implementations own
// their target statistics, matrix projection and whether the point has a
// free mean; the trainer owns topology, command submission and backtracking.
class CBMFullMatrixRuntime {
public:
    struct Selected { uint32_t Index; float Score,Gain; };
    virtual ~CBMFullMatrixRuntime() = default;
    virtual uint64_t AllocatedBytes() const = 0;
    virtual bool HasObjectiveValue() const { return true; }
    virtual double TakeAuxiliaryGPUSeconds() { return 0.0; }
    // Generated objectives sample a separate fixed target at each dataset's
    // original cursor. Deterministic objectives need no additional context.
    virtual void SetTargetPoint(id<MTLBuffer>, const CBMBootstrapOptions&, uint32_t, uint32_t) {}
    virtual void ClearStatus() = 0;
    virtual void CheckStatus() const = 0;
    virtual void EncodeEdges(id<MTLCommandBuffer> command,id<MTLBuffer> cursor,id<MTLBuffer> values,
        id<MTLBuffer> ids,uint32_t leaves,bool shifted,const CBMBootstrapOptions* bootstrap=nullptr,
        uint32_t absoluteIteration=0,uint64_t* dispatches=nullptr,bool trial=false) = 0;
    virtual void EncodeCandidates(id<MTLCommandBuffer> command,id<MTLBuffer> bins,id<MTLBuffer> ids,
        id<MTLBuffer> features,id<MTLBuffer> borders,id<MTLBuffer> types,uint32_t featureCount,
        uint32_t parents,bool gradientScore,float l2,float nonDiag,uint64_t* dispatches=nullptr,
        uint32_t firstCandidate=0,uint32_t candidateCount=0) = 0;
    virtual void EncodeSimpleLeafValues(id<MTLCommandBuffer> command,id<MTLBuffer> values,
        id<MTLBuffer> weights,uint32_t leaves,bool oneHot,uint64_t* dispatches=nullptr) = 0;
    virtual void EncodeSelectWinner(id<MTLCommandBuffer> command,id<MTLBuffer> features,id<MTLBuffer> featureWeights,
        uint32_t featureCount,float previousScore,bool packedWeights=false,uint64_t* dispatches=nullptr) = 0;
    virtual Selected ReadWinner() const = 0;
    virtual void EncodeLeafLayout(id<MTLCommandBuffer> command,id<MTLBuffer> ids,uint32_t leaves,uint64_t* dispatches=nullptr) = 0;
    virtual void EncodeLeafProjection(id<MTLCommandBuffer> command,uint32_t leaves,bool gradientMethod,uint64_t* dispatches=nullptr) = 0;
    virtual void EncodeLeafWeights(id<MTLCommandBuffer> command,id<MTLBuffer> originalWeights,id<MTLBuffer> rows,
        id<MTLBuffer> offsets,id<MTLBuffer> result,uint32_t leaves,uint64_t* dispatches=nullptr) = 0;
    virtual void EncodeLeafDirection(id<MTLCommandBuffer> command,uint32_t leaves,float l2,float nonDiag,uint64_t* dispatches=nullptr) = 0;
    virtual void EncodeLeafUpdate(id<MTLCommandBuffer> command,id<MTLBuffer> point,id<MTLBuffer> weights,
        id<MTLBuffer> updated,uint32_t leaves,float step,uint64_t* dispatches=nullptr,bool trial=false) = 0;
    virtual void EncodeBeginTrial(id<MTLCommandBuffer> command) = 0;
    virtual void EncodeDirectionDot(id<MTLCommandBuffer> command,id<MTLBuffer> output,uint32_t leaves,uint64_t* dispatches=nullptr) = 0;
    virtual double ReadTrialLoss() const = 0;
    virtual void EncodeCenterSolvedPoint(id<MTLCommandBuffer> command,id<MTLBuffer> point,uint32_t leaves,uint64_t* dispatches=nullptr) = 0;
    virtual void EncodeLoss(id<MTLCommandBuffer> command,id<MTLBuffer> ids,uint32_t leaves,bool supportOnly,uint64_t* dispatches=nullptr) = 0;
    virtual std::pair<double,double> ReadLoss() const = 0;
};
