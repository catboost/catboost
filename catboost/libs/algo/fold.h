#pragma once

#include "target_classifier.h"

#include "online_ctr.h"
#include "approx_util.h"
#include "projection.h"

#include <catboost/libs/model/online_ctr.h>
#include <catboost/libs/helpers/clear_array.h>
#include <catboost/libs/data/pair.h>
#include <catboost/libs/options/defaults_helper.h>

#include <util/generic/vector.h>
#include <util/random/shuffle.h>
#include <util/generic/ymath.h>

#include <tuple>

TVector<int> InvertPermutation(const TVector<int>& permutation);

int UpdateSizeForQueries(int size, const TVector<int>& queriesFinishIndex);

int SelectMinBatchSize(const int sampleCount, const TVector<int>& queriesFinishIndex);

double SelectTailSize(const int oldSize, const double multiplier, const TVector<int>& queriesFinishIndex);

struct TFold {
    struct TBodyTail {
        TVector<TVector<double>> Approx;
        TVector<TVector<double>> Derivatives;
        // TODO(annaveronika): make a single vector<vector> for all BodyTail
        TVector<TVector<double>> WeightedDer;
        TVector<TVector<TCompetitor>> Competitors;

        int BodyFinish = 0;
        int TailFinish = 0;
    };

    TVector<float> LearnWeights;
    TVector<ui32> LearnQueryId;
    THashMap<ui32, ui32> LearnQuerySize;
    TVector<int> LearnPermutation; // index in original array
    TVector<TBodyTail> BodyTailArr;
    TVector<float> LearnTarget;
    TVector<float> SampleWeights;
    TVector<TVector<int>> LearnTargetClass;
    TVector<int> TargetClassesCount;
    size_t EffectiveDocCount = 0;
    int PermutationBlockSize = FoldPermutationBlockSizeNotSet;

    TOnlineCTRHash& GetCtrs(const TProjection& proj) {
        return proj.HasSingleFeature() ? OnlineSingleCtrs : OnlineCTR;
    }

    const TOnlineCTRHash& GetCtrs(const TProjection& proj) const {
        return proj.HasSingleFeature() ? OnlineSingleCtrs : OnlineCTR;
    }

    TOnlineCTR& GetCtrRef(const TProjection& proj) {
        return GetCtrs(proj)[proj];
    }

    const TOnlineCTR& GetCtr(const TProjection& proj) const {
        return GetCtrs(proj).at(proj);
    }

    void DropEmptyCTRs();

    const std::tuple<const TOnlineCTRHash&, const TOnlineCTRHash&> GetAllCtrs() const {
        return std::tie(OnlineSingleCtrs, OnlineCTR);
    }

    void AssignTarget(const TVector<float>& target,
                      const TVector<TTargetClassifier>& targetClassifiers);

    template <typename T>
    void AssignPermuted(const TVector<T>& source, TVector<T>* dest) const {
        int learnSampleCount = LearnPermutation.ysize();
        TVector<T>& destination = *dest;
        destination.yresize(learnSampleCount);
        for (int z = 0; z < learnSampleCount; ++z) {
            int i = LearnPermutation[z];
            destination[z] = source[i];
        }
    }

    void AssignCompetitors(const TVector<TPair>& pairs,
                           const TVector<int>& invertPermutation,
                           TFold::TBodyTail* bt);

    int GetApproxDimension() const {
        return BodyTailArr[0].Approx.ysize();
    }

    void TrimOnlineCTR(size_t maxOnlineCTRFeatures) {
        if (OnlineCTR.size() > maxOnlineCTRFeatures) {
            OnlineCTR.clear();
        }
    }

    void SaveApproxes(IOutputStream* s) const;
    void LoadApproxes(IInputStream* s);
private:
    TOnlineCTRHash OnlineSingleCtrs;
    TOnlineCTRHash OnlineCTR;
};

void InitFromBaseline(const int beginIdx, const int endIdx,
                        const TVector<TVector<double>>& baseline,
                        const TVector<int>& learnPermutation,
                        bool storeExpApproxes,
                        TVector<TVector<double>>* approx);

TVector<int> CalcQueriesFinishIndex(const TVector<ui32>& queriesId);

struct TRestorableFastRng64;
class TTrainData;

TFold BuildLearnFold(const TTrainData& data,
                            const TVector<TTargetClassifier>& targetClassifiers,
                            bool shuffle,
                            int permuteBlockSize,
                            int approxDimension,
                            double multiplier,
                            bool storeExpApproxes,
                            TRestorableFastRng64& rand);

TFold BuildAveragingFold(const TTrainData& data,
                                const TVector<TTargetClassifier>& targetClassifiers,
                                bool shuffle,
                                int approxDimension,
                                bool storeExpApproxes,
                                TRestorableFastRng64& rand);
