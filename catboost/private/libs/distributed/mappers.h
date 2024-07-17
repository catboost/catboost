#pragma once

#include "data_types.h"

#include <catboost/libs/helpers/vector_helpers.h>
#include <catboost/private/libs/algo/tensor_search_helpers.h>

#include <library/cpp/containers/2d_array/2d_array.h>
#include <library/cpp/par/par.h>
#include <library/cpp/par/par_util.h>

#include <util/ysafeptr.h>


namespace NCatboostDistributed {

    class TDatasetsLoader: public NPar::TMapReduceCmd<TDatasetLoaderParams, TUnusedInitializedParam> {
        OBJECT_NOCOPY_METHODS(TDatasetsLoader);
        void DoMap(NPar::IUserContext* ctx, int hostId, TInput* params, TOutput* /*unused*/) const final;
    };
    class TPlainFoldBuilder: public NPar::TMapReduceCmd<TPlainFoldBuilderParams, TUnusedInitializedParam> {
        OBJECT_NOCOPY_METHODS(TPlainFoldBuilder);
        void DoMap(NPar::IUserContext* ctx, int hostId, TInput* params, TOutput* /*unused*/) const final;
    };
    class TApproxReconstructor
        : public NPar::TMapReduceCmd<TApproxReconstructorParams, TUnusedInitializedParam> {

        OBJECT_NOCOPY_METHODS(TApproxReconstructor);
        void DoMap(NPar::IUserContext* ctx, int hostId, TInput* params, TOutput* /*unused*/) const final;
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
    class TDerivativesStDevFromZeroCalcer: public NPar::TMapReduceCmd<TUnusedInitializedParam, double> {
        OBJECT_NOCOPY_METHODS(TDerivativesStDevFromZeroCalcer);
        void DoMap(NPar::IUserContext* ctx, int hostId, TInput* /*unused*/, TOutput* /*unused*/) const final;
    };

    // [cand][subcand]
    class TScoreCalcer: public NPar::TMapReduceCmd<TCandidateList, TStats5D> {
        OBJECT_NOCOPY_METHODS(TScoreCalcer);
        void DoMap(
            NPar::IUserContext* ctx,
            int hostId,
            TInput* candidateList,
            TOutput* bucketStats) const final;
    };

    // [cand][subcand]
    class TPairwiseScoreCalcer:
        public NPar::TMapReduceCmd<TCandidateList, TWorkerPairwiseStats> {

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
    class TLeafIndexSetter: public NPar::TMapReduceCmd<TSplit, TUnusedInitializedParam> {
        OBJECT_NOCOPY_METHODS(TLeafIndexSetter);
        void DoMap(
            NPar::IUserContext* ctx,
            int hostId,
            TInput* bestSplit,
            TOutput* /*unused*/) const final;
    };
    class TEmptyLeafFinder: public NPar::TMapReduceCmd<TUnusedInitializedParam, TIsLeafEmpty> {
        OBJECT_NOCOPY_METHODS(TEmptyLeafFinder);
        void DoMap(
            NPar::IUserContext* /*ctx*/,
            int /*hostId*/,
            TInput* /*unused*/,
            TOutput* isLeafEmpty) const final;
    };
    class TBucketSimpleUpdater:
        public NPar::TMapReduceCmd<TUnusedInitializedParam, std::pair<TSums, TArray2D<double>>> {

        OBJECT_NOCOPY_METHODS(TBucketSimpleUpdater);
        void DoMap(NPar::IUserContext* /*ctx*/, int /*hostId*/, TInput* /*unused*/, TOutput* sums) const final;
    };
    class TCalcApproxStarter: public NPar::TMapReduceCmd<std::variant<TSplitTree, TNonSymmetricTreeStructure>, TUnusedInitializedParam> {
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
    class TBestApproxSetter: public NPar::TMapReduceCmd<TUnusedInitializedParam, TUnusedInitializedParam> {
        OBJECT_NOCOPY_METHODS(TBestApproxSetter);
        void DoMap(
            NPar::IUserContext* /*ctx*/,
            int /*hostId*/,
            TInput* /*unused*/,
            TOutput* /*unused*/) const final;
    };
    class TApproxGetter: public NPar::TMapReduceCmd<TApproxGetterParams, TApproxesResult> {
        OBJECT_NOCOPY_METHODS(TApproxGetter);
        void DoMap(
            NPar::IUserContext* /*ctx*/,
            int /*hostId*/,
            TInput* approxGetterParams,
            TOutput* approxesResult) const final;
    };

    class TBucketMultiUpdater:
        public NPar::TMapReduceCmd<
            TUnusedInitializedParam,
            std::pair<TMultiSums, TUnusedInitializedParam>> {

        OBJECT_NOCOPY_METHODS(TBucketMultiUpdater);
        void DoMap(NPar::IUserContext* /*ctx*/, int /*hostId*/, TInput* /*unused*/, TOutput* sums) const final;
    };
    class TDeltaMultiUpdater: public NPar::TMapReduceCmd<TVector<TVector<double>>, TUnusedInitializedParam> {
        OBJECT_NOCOPY_METHODS(TDeltaMultiUpdater);
        void DoMap(NPar::IUserContext* ctx, int hostId, TInput* leafValues, TOutput* /*unused*/) const final;
    };
    class TErrorCalcer: public NPar::TMapReduceCmd<TErrorCalcerParams, TVector<THashMap<TString, TMetricHolder>>> {
        OBJECT_NOCOPY_METHODS(TErrorCalcer);
        void DoMap(NPar::IUserContext* ctx, int hostId, TInput* params, TOutput* additiveStats) const final;
    };
    class TArmijoStartPointBackupper: public NPar::TMapReduceCmd<bool, TUnusedInitializedParam> {
        OBJECT_NOCOPY_METHODS(TArmijoStartPointBackupper);
        void DoMap(NPar::IUserContext* ctx, int hostId, TInput* isRestore, TOutput* /*unused*/) const final;
    };
    class TLeafWeightsGetter: public NPar::TMapReduceCmd<TUnusedInitializedParam, TVector<double>> {
        OBJECT_NOCOPY_METHODS(TLeafWeightsGetter);
        void DoMap(NPar::IUserContext* ctx, int hostId, TInput* /*unused*/, TOutput* leafWeights) const final;
        void DoReduce(TVector<TOutput>* leafWeightsFromWorkers, TOutput* totalLeafWeights) const final;
    };
    class TQuantileExactApproxStarter: public NPar::TMapReduceCmd<TUnusedInitializedParam, TVector<TVector<TMinMax<double>>>> {
        OBJECT_NOCOPY_METHODS(TQuantileExactApproxStarter);
        void DoMap(NPar::IUserContext* ctx, int hostId, TInput* leafCount, TOutput* minMaxDiffs) const final;
        void DoReduce(TVector<TOutput>* minMaxDiffsFromWorkers, TOutput* reducedMinMaxDiffs) const final;
    };
    class TQuantileArraySplitter: public NPar::TMapReduceCmd<TVector<TVector<double>>, TVector<TVector<double>>> {
        OBJECT_NOCOPY_METHODS(TQuantileArraySplitter);
        void DoMap(NPar::IUserContext* ctx, int hostId, TInput* pivots, TOutput* leftSumWeights) const final;
        void DoReduce(TVector<TOutput>* leftRightWeightsFromWorkers, TOutput* totalLeftSumWeights) const final;
    };
    class TQuantileEqualWeightsCalcer: public NPar::TMapReduceCmd<TVector<TVector<double>>, TVector<TVector<double>>> {
        OBJECT_NOCOPY_METHODS(TQuantileEqualWeightsCalcer);
        void DoMap(NPar::IUserContext* ctx, int hostId, TInput* pivots, TOutput* equalSumWeights) const final;
        void DoReduce(TVector<TOutput>* equalSumWeightsFromWorkers, TOutput* totalEqualSumWeights) const final;
    };

} // NCatboostDistributed
