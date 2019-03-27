#pragma once

#include <catboost/libs/helpers/exception.h>
#include <library/containers/dense_hash/dense_hash.h>
#include <util/generic/array_ref.h>
#include <util/generic/ptr.h>
#include <util/generic/string.h>


namespace NCB {

    struct TTokenId {
        ui32 Id;

        TTokenId()
        : Id(static_cast<ui32>(-1)) {

        }

        TTokenId(ui32 id)
        : Id(id) {

        }


        operator ui32() const {
            return Id;
        }

        bool operator==(const TTokenId& rhs) const {
            return Id == rhs.Id;
        }
        bool operator!=(const TTokenId& rhs) const {
            return !(rhs == *this);
        }

        bool operator<(const TTokenId& rhs) const {
            return Id < rhs.Id;
        }
        bool operator>(const TTokenId& rhs) const {
            return rhs < *this;
        }
        bool operator<=(const TTokenId& rhs) const {
            return !(rhs < *this);
        }
        bool operator>=(const TTokenId& rhs) const {
            return !(*this < rhs);
        }
    };

    using TText = TDenseHash<TTokenId, ui32>;


    class TTextDataSet : public TThrRefBase {
    public:
        TTextDataSet(TVector<TText> texts)
        : Text(std::move(texts)) {

        }

        ui64 SamplesCount() const {
            return Text.size();
        }

        const TText& GetText(ui64 idx) const {
            CB_ENSURE(idx < Text.size(), "Error: text line " << idx << " is out of bound (" << Text.size() << ")");
            return Text[idx];
        }

        TConstArrayRef<TText> GetTexts() const {
            return MakeConstArrayRef(Text);
        }
    private:
        TVector<TText> Text;
    };


    using TTextDataSetPtr = TIntrusivePtr<TTextDataSet>;

    struct TTextClassificationTarget : public TThrRefBase {
        TVector<ui32> Classes;
        ui32 NumClasses;
    };


    using TTextClassificationTargetPtr = TIntrusivePtr<TTextClassificationTarget>;


}
