#pragma once
#include "data_types.h"

#include <catboost/libs/algo/tensor_search_helpers.h>

#include <library/par/par.h>
#include <library/par/par_util.h>

#include <util/ysafeptr.h>

namespace NCatboostDistributed {
struct TUnusedInitializedParam {
    TUnusedInitializedParam() : Zero(0) {}
private:
    char Zero;
};
class TPlainFoldBuilder: public NPar::TMapReduceCmd<TUnusedInitializedParam, TUnusedInitializedParam> {
    OBJECT_NOCOPY_METHODS(TPlainFoldBuilder);
    void DoMap(NPar::IUserContext* ctx, int hostId, TInput* /*unused*/, TOutput* /*unused*/) const final;
};
class TTensorSearchStarter: public NPar::TMapReduceCmd<TUnusedInitializedParam, TUnusedInitializedParam> {
    OBJECT_NOCOPY_METHODS(TTensorSearchStarter);
    void DoMap(NPar::IUserContext* /*ctx*/, int /*hostId*/, TInput* /*unused*/, TOutput* /*unused*/) const final;
};
class TBootstrapMaker: public NPar::TMapReduceCmd<TUnusedInitializedParam, TUnusedInitializedParam> {
    OBJECT_NOCOPY_METHODS(TBootstrapMaker);
    void DoMap(NPar::IUserContext* ctx, int hostId, TInput* /*unused*/, TOutput* /*unused*/) const final;
};
class TScoreCalcer: public NPar::TMapReduceCmd<TEnvelope<TCandidateList>, TEnvelope<TStats5D>> { // [cand][subcand][bodytail + dim][leaf][bucket]
    OBJECT_NOCOPY_METHODS(TScoreCalcer);
    void DoMap(NPar::IUserContext* ctx, int hostId, TInput* candidateList, TOutput* bucketStats) const final;
};
class TRemoteBinCalcer: public NPar::TMapReduceCmd<TCandidatesInfoList, TStats4D> { // [subcand][bodytail + dim][leaf][bucket]
    OBJECT_NOCOPY_METHODS(TRemoteBinCalcer);
    void DoMap(NPar::IUserContext* ctx, int hostId, TInput* cadidate, TOutput* bucketStats) const final;
    void DoReduce(TVector<TOutput>* bucketStatsFromAllWorkers, TOutput* bucketStats) const final;
};
class TRemoteScoreCalcer: public NPar::TMapReduceCmd<TStats4D, TVector<TVector<double>>> {
    OBJECT_NOCOPY_METHODS(TRemoteScoreCalcer);
    void DoMap(NPar::IUserContext* ctx, int hostId, TInput* bucketStats, TOutput* scores) const final;
};
class TLeafIndexSetter: public NPar::TMapReduceCmd<TEnvelope<TCandidateInfo>, TUnusedInitializedParam> {
    OBJECT_NOCOPY_METHODS(TLeafIndexSetter);
    void DoMap(NPar::IUserContext* ctx, int hostId, TInput* bestSplitCandidate, TOutput* /*unused*/) const final;
};
class TEmptyLeafFinder: public NPar::TMapReduceCmd<TUnusedInitializedParam, TEnvelope<TIsLeafEmpty>> {
    OBJECT_NOCOPY_METHODS(TEmptyLeafFinder);
    void DoMap(NPar::IUserContext* /*ctx*/, int /*hostId*/, TInput* /*unused*/, TOutput* isLeafEmpty) const final;
};
template<typename TError>
class TBucketSimpleUpdater: public NPar::TMapReduceCmd<TUnusedInitializedParam, TEnvelope<TSums>> {
    OBJECT_NOCOPY_METHODS(TBucketSimpleUpdater);
    void DoMap(NPar::IUserContext* /*ctx*/, int /*hostId*/, TInput* /*unused*/, TOutput* sums) const final;
};
class TCalcApproxStarter: public NPar::TMapReduceCmd<TEnvelope<TSplitTree>, TUnusedInitializedParam> {
    OBJECT_NOCOPY_METHODS(TCalcApproxStarter);
    void DoMap(NPar::IUserContext* ctx, int hostId, TInput* splitTree, TOutput* /*unused*/) const final;
};
class TDeltaSimpleUpdater: public NPar::TMapReduceCmd<TEnvelope<TSums>, TUnusedInitializedParam> {
    OBJECT_NOCOPY_METHODS(TDeltaSimpleUpdater);
    void DoMap(NPar::IUserContext* ctx, int hostId, TInput* sums, TOutput* /*unused*/) const final;
};
class TApproxSimpleUpdater: public NPar::TMapReduceCmd<TUnusedInitializedParam, TUnusedInitializedParam> {
    OBJECT_NOCOPY_METHODS(TApproxSimpleUpdater);
    void DoMap(NPar::IUserContext* ctx, int hostId, TInput* /*unused*/, TOutput* /*unused*/) const final;
};
template<typename TError>
class TDerivativeSetter: public NPar::TMapReduceCmd<TUnusedInitializedParam, TUnusedInitializedParam> {
    OBJECT_NOCOPY_METHODS(TDerivativeSetter);
    void DoMap(NPar::IUserContext* /*ctx*/, int /*hostId*/, TInput* /*unused*/, TOutput* /*unused*/) const final;
};
} // NCatboostDistributed
