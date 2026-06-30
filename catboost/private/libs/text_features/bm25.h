#pragma once

#include "feature_calcer.h"

#include <library/cpp/containers/dense_hash/dense_hash.h>
#include <util/system/types.h>
#include <util/generic/fwd.h>

namespace NCB {

    /*
     *  BM25 between class and text documents
     *  Convert to classical: class = document (BoW)
     *  query = text
     *  BM25(class, text)
    */
    class TBM25 final : public TTextFeatureCalcer {
    public:

        explicit TBM25(
            const TGuid& calcerId = CreateGuid(),
            ui32 numClasses = 2,
            double truncateBorder = 1e-3,
            double k = 1.5,
            double b = 0.75
        );

        bool IsSerializable() const override {
            return true;
        }

        EFeatureCalcerType Type() const override {
            return EFeatureCalcerType::BM25;
        }

        void Compute(const TText& text, TOutputFloatIterator iterator) const override;

        static ui32 BaseFeatureCount(ui32 numClasses) {
            return numClasses;
        }

    private:
        ui32 NumClasses;
        double K;
        double B;
        double TruncateBorder;

        ui64 TotalTokens;
        TVector<ui64> ClassTotalTokens;
        TVector<TDenseHash<TTokenId, ui32>> Frequencies;
        TVector<double> TruncatedInvClassFreq;

    protected:
        TTextFeatureCalcer::TFeatureCalcerFbs SaveParametersToFB(flatbuffers::FlatBufferBuilder& builder) const override;
        void LoadParametersFromFB(const NCatBoostFbs::TFeatureCalcer* calcerFbs) override;

        void SaveLargeParameters(IOutputStream* ) const override;
        void LoadLargeParameters(IInputStream* ) override;

        friend class TBM25Visitor;
    };

    class TBM25Visitor final : public ITextCalcerVisitor {
    public:
        void Update(ui32 classId, const TText& text, TTextFeatureCalcer* bm25) override;
    };
}
