#pragma once

#include <catboost/libs/algo/learn_context.h>
#include <catboost/libs/algo/split.h>
#include <catboost/libs/algo/tensor_search_helpers.h>
#include <catboost/libs/algo/dataset.h>

void InitializeMaster(TLearnContext* ctx);
void FinalizeMaster(TLearnContext* ctx);
void MapBuildPlainFold(const TDataset& trainData, TLearnContext* ctx);
void MapTensorSearchStart(TLearnContext* ctx);
void MapBootstrap(TLearnContext* ctx);
void MapCalcScore(double scoreStDev, int depth, TCandidateList* candidateList, TLearnContext* ctx);
void MapRemoteCalcScore(double scoreStDev, int depth, TCandidateList* candidateList, TLearnContext* ctx);
void MapSetIndices(const TCandidateInfo& bestSplitCandidate, TLearnContext* ctx);
int MapGetRedundantSplitIdx(TLearnContext* ctx);
template<typename TError>
void MapSetDerivatives(TLearnContext* ctx);
template<typename TError>
void MapSetApproxes(const TSplitTree& splitTree, TLearnContext* ctx);

