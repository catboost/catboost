#pragma once

#include "data_types.h"

#include <catboost/libs/algo/tensor_search_helpers.h>

#include <library/par/par.h>
#include <library/par/par_util.h>

#include <util/ysafeptr.h>


namespace NCatboostDistributed {

    class TPlainFoldBuilder: public NPar::TMapReduceCmd<TUnusedInitializedParam, TUnusedInitializedParam> {
        OBJECT_NOCOPY_METHODS(TPlainFoldBuilder);
        void DoMap(NPar::IUserContext* ctx, int hostId, TInput* /*unused*/, TOutput* /*unused*/) const final;
    };
    class TApproxReconstructor
        : public NPar::TMapReduceCmd<
          TEnvelope<std::pair<TVector<TSplitTree>, TVector<TVector<TVector<double>>>>>,
          TUnusedInitializedParam> {

        OBJECT_NOCOPY_METHODS(TApproxReconstructor);
        void DoMap(NPar::IUserContext* ctx, int hostId, TInput* forest, TOutput* /*unused*/) const final;
    };
    class TTensorSearchStarter: public NPar::TMapReduceCmd<TUnusedInitializedParam, TUnusedInitializedParam> {
        OBJECT_NOCOPY_METHODS(TTensorSearchStarter);
        void DoMap(
            NPar::IUserContext* /*ctx*/,
            int /*hostId*/,
            TInput* /*unused*/,
            TOutput* /*unused*/) const final;
    };
    class TBootstrapMaker: public NPar::TMapReduceCmd<TUnusedInitializedParam, TUnusedInitializedParam> {
        OBJECT_NOCOPY_METHODS(TBootstrapMaker);
        void DoMap(NPar::IUserContext* ctx, int hostId, TInput* /*unused*/, TOutput* /*unused*/) const final;
    };

    // [cand][subcand]
    class TScoreCalcer: public NPar::TMapReduceCmd<TEnvelope<TCandidateList>, TEnvelope<TStats5D>> {
        OBJECT_NOCOPY_METHODS(TScoreCalcer);
        void DoMap(
            NPar::IUserContext* ctx,
            int hostId,
            TInput* candidateList,
            TOutput* bucketStats) const final;
    };

    // [cand][subcand]
    class TPairwiseScoreCalcer:
        public NPar::TMapReduceCmd<TEnvelope<TCandidateList>, TEnvelope<TWorkerPairwiseStats>> {

        OBJECT_NOCOPY_METHODS(TPairwiseScoreCalcer);
        void DoMap(
            NPar::IUserContext* ctx,
            int hostId,
            TInput* candidateList,
            TOutput* bucketStats) const final;
    };

    // [cand]
    class TRemotePairwiseBinCalcer: public NPar::TMapReduceCmd<TCandidatesInfoList, TVector<TPairwiseStats>> {
        OBJECT_NOCOPY_METHODS(TRemotePairwiseBinCalcer);
        void DoMap(
            NPar::IUserContext* ctx,
            int hostId,
            TInput* subcandidates,
            TOutput* bucketStats) const final;
        void DoReduce(TVector<TOutput>* statsFromAllWorkers, TOutput* bucketStats) const final;
    };
    class TRemotePairwiseScoreCalcer:
        public NPar::TMapReduceCmd<TVector<TPairwiseStats>, TVector<TVector<double>>> {

        OBJECT_NOCOPY_METHODS(TRemotePairwiseScoreCalcer);
        void DoMap(NPar::IUserContext* ctx, int hostId, TInput* bucketStats, TOutput* scores) const final;
    };
    class TRemoteBinCalcer: public NPar::TMapReduceCmd<TCandidatesInfoList, TStats4D> { // [subcand]
        OBJECT_NOCOPY_METHODS(TRemoteBinCalcer);
        void DoMap(NPar::IUserContext* ctx, int hostId, TInput* candidatesInfoList, TOutput* bucketStats) const final;
        void DoReduce(TVector<TOutput>* statsFromAllWorkers, TOutput* bucketStats) const final;
    };
    class TRemoteScoreCalcer: public NPar::TMapReduceCmd<TStats4D, TVector<TVector<double>>> {
        OBJECT_NOCOPY_METHODS(TRemoteScoreCalcer);
        void DoMap(NPar::IUserContext* ctx, int hostId, TInput* bucketStats, TOutput* scores) const final;
    };
    class TLeafIndexSetter: public NPar::TMapReduceCmd<TEnvelope<TSplit>, TUnusedInitializedParam> {
        OBJECT_NOCOPY_METHODS(TLeafIndexSetter);
        void DoMap(
            NPar::IUserContext* ctx,
            int hostId,
            TInput* bestSplit,
            TOutput* /*unused*/) const final;
    };
    class TEmptyLeafFinder: public NPar::TMapReduceCmd<TUnusedInitializedParam, TEnvelope<TIsLeafEmpty>> {
        OBJECT_NOCOPY_METHODS(TEmptyLeafFinder);
        void DoMap(
            NPar::IUserContext* /*ctx*/,
            int /*hostId*/,
            TInput* /*unused*/,
            TOutput* isLeafEmpty) const final;
    };
    class TBucketSimpleUpdater:
        public NPar::TMapReduceCmd<TUnusedInitializedParam, TEnvelope<std::pair<TSums, TArray2D<double>>>> {

        OBJECT_NOCOPY_METHODS(TBucketSimpleUpdater);
        void DoMap(NPar::IUserContext* /*ctx*/, int /*hostId*/, TInput* /*unused*/, TOutput* sums) const final;
    };
    class TCalcApproxStarter: public NPar::TMapReduceCmd<TEnvelope<TSplitTree>, TUnusedInitializedParam> {
        OBJECT_NOCOPY_METHODS(TCalcApproxStarter);
        void DoMap(NPar::IUserContext* ctx, int hostId, TInput* splitTree, TOutput* /*unused*/) const final;
    };
    class TDeltaSimpleUpdater: public NPar::TMapReduceCmd<TVector<TVector<double>>, TUnusedInitializedParam> {
        OBJECT_NOCOPY_METHODS(TDeltaSimpleUpdater);
        void DoMap(NPar::IUserContext* ctx, int hostId, TInput* sums, TOutput* /*unused*/) const final;
    };
    class TApproxUpdater: public NPar::TMapReduceCmd<TVector<TVector<double>>, TUnusedInitializedParam> {
        OBJECT_NOCOPY_METHODS(TApproxUpdater);
        void DoMap(
            NPar::IUserContext* ctx,
            int hostId,
            TInput* averageLeafValues,
            TOutput* /*unused*/) const final;
    };
    class TDerivativeSetter: public NPar::TMapReduceCmd<TUnusedInitializedParam, TUnusedInitializedParam> {
        OBJECT_NOCOPY_METHODS(TDerivativeSetter);
        void DoMap(
            NPar::IUserContext* /*ctx*/,
            int /*hostId*/,
            TInput* /*unused*/,
            TOutput* /*unused*/) const final;
    };
    class TBucketMultiUpdater:
        public NPar::TMapReduceCmd<
            TUnusedInitializedParam,
            TEnvelope<std::pair<TMultiSums, TUnusedInitializedParam>>> {

        OBJECT_NOCOPY_METHODS(TBucketMultiUpdater);
        void DoMap(NPar::IUserContext* /*ctx*/, int /*hostId*/, TInput* /*unused*/, TOutput* sums) const final;
    };
    class TDeltaMultiUpdater: public NPar::TMapReduceCmd<TVector<TVector<double>>, TUnusedInitializedParam> {
        OBJECT_NOCOPY_METHODS(TDeltaMultiUpdater);
        void DoMap(NPar::IUserContext* ctx, int hostId, TInput* leafValues, TOutput* /*unused*/) const final;
    };
    class TErrorCalcer: public NPar::TMapReduceCmd<TUnusedInitializedParam, THashMap<TString, TMetricHolder>> {
        OBJECT_NOCOPY_METHODS(TErrorCalcer);
        void DoMap(NPar::IUserContext* ctx, int hostId, TInput* /*unused*/, TOutput* additiveStats) const final;
    };
    class TLeafWeightsGetter: public NPar::TMapReduceCmd<TUnusedInitializedParam, TVector<double>> {
        OBJECT_NOCOPY_METHODS(TLeafWeightsGetter);
        void DoMap(NPar::IUserContext* ctx, int hostId, TInput* /*unused*/, TOutput* leafWeights) const final;
    };

} // NCatboostDistributed
