#include "binarization.h"
#include <util/generic/queue.h>
#include <util/generic/algorithm.h>
#include <util/generic/ymath.h>
#include <util/generic/yexception.h>

namespace NSplitSelection {
class TFeatureBin {
private:
    ui32 BinStart;
    ui32 BinEnd;
    yvector<float>::const_iterator FeaturesStart;
    yvector<float>::const_iterator FeaturesEnd;

    ui32 BestSplit;
    double BestScore;

    inline void UpdateBestSplitProperties() {
        const int mid = (BinStart + BinEnd) / 2;
        float midValue = *(FeaturesStart + mid);

        ui32 lb = (ui32)(LowerBound(FeaturesStart + BinStart, FeaturesStart + mid, midValue) - FeaturesStart);
        ui32 up = (ui32)(UpperBound(FeaturesStart + mid, FeaturesStart + BinEnd, midValue) - FeaturesStart);

        const double scoreLeft = lb != BinStart ? log((double)(lb - BinStart)) + log((double)(BinEnd - lb)) : 0.0;
        const double scoreRight = up != BinEnd ? log((double)(up - BinStart)) + log((double)(BinEnd - up)) : 0.0;
        BestSplit = scoreLeft >= scoreRight ? lb : up;
        BestScore = BestSplit == lb ? scoreLeft : scoreRight;
    }

public:
    TFeatureBin(ui32 binStart, ui32 binEnd, const yvector<float>::const_iterator featuresStart, const yvector<float>::const_iterator featuresEnd)
        : BinStart(binStart)
        , BinEnd(binEnd)
        , FeaturesStart(featuresStart)
        , FeaturesEnd(featuresEnd)
        , BestSplit(BinStart)
        , BestScore(0.0)
    {
        UpdateBestSplitProperties();
    }

    ui32 Size() const {
        return BinEnd - BinStart;
    }

    bool operator<(const TFeatureBin& bf) const {
        return Score() < bf.Score();
    }

    double Score() const {
        return BestScore;
    }

    TFeatureBin Split() {
        if (!CanSplit()) {
            throw yexception() << "Can't add new split";
        }
        TFeatureBin left = TFeatureBin(BinStart, BestSplit, FeaturesStart, FeaturesEnd);
        BinStart = BestSplit;
        UpdateBestSplitProperties();
        return left;
    }

    bool CanSplit() const {
        return (BinStart != BestSplit && BinEnd != BestSplit);
    }

    float Border() const {
        Y_ASSERT(BinStart < BinEnd);
        float borderValue = 0.5f * (*(FeaturesStart + BinEnd - 1));
        const float nextValue = ((FeaturesStart + BinEnd) < FeaturesEnd)
                                ? (*(FeaturesStart + BinEnd))
                                : (*(FeaturesStart + BinEnd - 1));
        borderValue += 0.5f * nextValue;
        return borderValue;
    }

    bool IsLast() const {
        return BinEnd == (FeaturesEnd - FeaturesStart);
    }
};

yhash_set<float> TMedianInBinBinarizer::BestSplit(yvector<float>& featureValues,
                                                  int bordersCount, bool isSorted) const {

    if (!isSorted) {
        Sort(featureValues.begin(), featureValues.end());
    }

    std::priority_queue<TFeatureBin> splits;
    splits.push(TFeatureBin(0, (ui32)featureValues.size(), featureValues.begin(), featureValues.end()));

    while (splits.size() <= (ui32)bordersCount && splits.top().CanSplit()) {
        TFeatureBin top = splits.top();
        splits.pop();
        splits.push(top.Split());
        splits.push(top);
    }

    yhash_set<float> borders;
    while (splits.size()) {
        if (!splits.top().IsLast())
            borders.insert(splits.top().Border());
        splits.pop();
    }
    return borders;
}
}
