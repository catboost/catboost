#include "embedding.h"

using namespace NCB;

class TEmbedding final : public IEmbedding {
public:
    explicit TEmbedding(TDenseHash<TTokenId, TVector<float>>&& embedding)
    : Embedding(std::move(embedding)) {

    }

    ui64 Dim() const override {
        return Embedding.begin()->second.size();
    }


    void Apply(const TTextDataSet& ds, TVector<TVector<float>>* dst, NPar::TLocalExecutor* executor) const override {
        dst->resize(ds.SamplesCount());
        auto texts = ds.GetTexts();
        NPar::ParallelFor(*executor, 0, static_cast<ui32>(texts.size()), [&](ui32 idx) {
            Apply(texts[idx], &(*dst)[idx]);
        });
    }


    void Apply(const TText& text, TVector<float>* dst) const {
        dst->clear();
        dst->resize(Dim());
        auto& result = *dst;
        double count = 0.5;
        for (const auto& tokenToCount : text) {
            auto embedding = Embedding.find(tokenToCount.Token());
            if (embedding != Embedding.end()) {
                AddVector(embedding->second, result);
                ++count;
            }
        }
        for (ui64 i = 0; i < result.size(); ++i) {
            result[i] /= count;
        }
    }
private:

    void AddVector(TConstArrayRef<float> what, TArrayRef<float> to) const {
        for (ui32 i = 0; i < to.size(); ++i) {
            to[i] += what[i];
        }
    }
private:
    TDenseHash<TTokenId, TVector<float>> Embedding;


};

TEmbeddingPtr NCB::CreateEmbedding(TDenseHash<TTokenId, TVector<float>>&& hash) {
    return new TEmbedding(std::move(hash));
}
