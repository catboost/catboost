#include "dictionary.h"

namespace NCB {
    TDictionaryProxy::TDictionaryProxy(TDictionaryPtr dictionaryImpl)
        : DictionaryImpl(std::move(dictionaryImpl))
        , Guid(CreateGuid())
    {}

    TGuid TDictionaryProxy::Id() const {
        return Guid;
    }

    void TDictionaryProxy::Apply(TConstArrayRef<TStringBuf> tokens, TText* text) const {
        text->Clear();

        TVector<ui32> tokenIds;
        DictionaryImpl->Apply(tokens, &tokenIds);
        for (const auto& tokenId : tokenIds) {
            (*text)[TTokenId(tokenId)]++;
        }
    }

    ui32 TDictionaryProxy::Size() const {
        return DictionaryImpl->Size();
    }

    TText TDictionaryProxy::Apply(TConstArrayRef<TStringBuf> tokens) const {
        TText result;
        Apply(tokens, &result);
        return result;
    }

    TTokenId TDictionaryProxy::Apply(TStringBuf token) const {
        return TTokenId(DictionaryImpl->Apply(token));
    }

    TTokenId TDictionaryProxy::GetUnknownTokenId() const {
        return TTokenId(DictionaryImpl->GetUnknownTokenId());
    }

    TVector<TTokenId> TDictionaryProxy::GetTopTokens(ui32 topSize) const {
        topSize = Min(topSize, DictionaryImpl->Size());
        return xrange(topSize);
    }

    void TDictionaryProxy::Save(IOutputStream* stream) const {
        WriteMagic(DictionaryMagic.data(), MagicSize, Alignment, stream);
        Guid.Save(stream);

        if (auto basicDictionary = dynamic_cast<TDictionary*>(DictionaryImpl.Get())) {
            TMMapDictionary mMapDictionary = TMMapDictionary(basicDictionary);
            mMapDictionary.Save(stream);
        } else if (auto mMapDictionary = dynamic_cast<TMMapDictionary*>(DictionaryImpl.Get())) {
            mMapDictionary->Save(stream);
        } else {
            CB_ENSURE(false, "Failed to serialize dictionary: Unknown dictionary type");
        }
    }

    void TDictionaryProxy::Load(IInputStream* stream) {
        ReadMagic(DictionaryMagic.data(), MagicSize, Alignment, stream);
        Guid.Load(stream);

        auto dictionaryImpl = MakeIntrusive<TMMapDictionary>();
        dictionaryImpl->Load(stream);
        DictionaryImpl = std::move(dictionaryImpl);
    }

}
