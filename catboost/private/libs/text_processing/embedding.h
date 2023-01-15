#pragma once

#include "text_dataset.h"

#include <library/cpp/containers/dense_hash/dense_hash.h>
#include <library/cpp/threading/local_executor/local_executor.h>

#include <util/system/types.h>
#include <util/generic/ptr.h>


namespace NCB {


    class IEmbedding : public TThrRefBase {
    public:

        virtual ui64 Dim() const = 0;

        virtual void Apply(const TTextDataSet& ds, TVector<TVector<float>>* dst, NPar::TLocalExecutor* executor) const = 0;

        virtual void Apply(const TText& text, TVector<float>* dst) const = 0;
    };

    using TEmbeddingPtr = TIntrusivePtr<IEmbedding>;


    TEmbeddingPtr CreateEmbedding(TDenseHash<TTokenId, TVector<float>>&& hash);
}


