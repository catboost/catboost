#pragma once

#include "feature_calcer.h"

#include <library/cpp/containers/dense_hash/dense_hash.h>
#include <util/system/types.h>
#include <util/generic/fwd.h>

namespace NCB {

    /*
    * p(x | c_k) = \prod p_i,k^{x_i} * (1 - p_{i, k})(1 - x_i), where x i binary vector
    */
    class TMultinomialNaiveBayes final : public TTextFeatureCalcer {
    public:
        static constexpr float DEFAULT_PRIOR = 0.5;
        static constexpr ui32 SEEN_TOKENS_PRIOR = 1;

        explicit TMultinomialNaiveBayes(
            const TGuid& calcerId = CreateGuid(),
            ui32 numClasses = 2,
            double classPrior = DEFAULT_PRIOR,
            double tokenPrior = DEFAULT_PRIOR,
            ui64 numSeenTokens = 0
        )
            : TTextFeatureCalcer(BaseFeatureCount(numClasses), calcerId)
            , NumClasses(numClasses)
            , ClassPrior(classPrior)
            , TokenPrior(tokenPrior)
            , NumSeenTokens(numSeenTokens)
            , ClassDocs(numClasses)
            , ClassTotalTokens(numClasses)
            , Frequencies(numClasses) {
        }

        void Compute(const TText& text, TOutputFloatIterator iterator) const override;

        static ui32 BaseFeatureCount(ui32 numClasses) {
            return numClasses > 2 ? numClasses : 1;
        }

        EFeatureCalcerType Type() const override {
            return EFeatureCalcerType::NaiveBayes;
        }

        bool IsSerializable() const override {
            return true;
        }

    private:
        double LogProb(
            const TDenseHash<TTokenId, ui32>& freqTable,
            double classSamples,
            double classTokensCount,
            const TText& text
        ) const;

    protected:
        TTextFeatureCalcer::TFeatureCalcerFbs SaveParametersToFB(flatbuffers::FlatBufferBuilder& builder) const override;
        void LoadParametersFromFB(const NCatBoostFbs::TFeatureCalcer* calcerFbs) override;

        void SaveLargeParameters(IOutputStream*) const override;
        void LoadLargeParameters(IInputStream*) override;

    private:
        ui32 NumClasses;
        double ClassPrior;
        double TokenPrior;

        ui64 NumSeenTokens;
        TVector<ui32> ClassDocs;
        TVector<ui64> ClassTotalTokens;
        TVector<TDenseHash<TTokenId, ui32>> Frequencies;

        friend class TNaiveBayesVisitor;
    };

    class TNaiveBayesVisitor final : public ITextCalcerVisitor {
    public:
        void Update(ui32 classId, const TText& text, TTextFeatureCalcer* naiveBayes) override;
    private:
        TDenseHashSet<ui32> SeenTokens;
    };
}
