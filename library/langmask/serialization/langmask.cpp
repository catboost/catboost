#include "langmask.h"

#include <library/langmask/proto/langmask.pb.h>

#include <util/string/vector.h>

void Serialize(NProto::TLangMask& message, const ::TLangMask& mask, bool humanReadable) {
    if (humanReadable) {
        message.SetNames(SerializeReadableLangMask(mask));
    } else {
        for (ELanguage bit : mask) {
            message.AddBits((ui32)bit);
        }
    }
}

TLangMask Deserialize(const NProto::TLangMask& message) {
    if (message.HasNames()) {
        return DeserializeReadableLangMask(message.GetNames());
    }

    TLangMask mask;
    size_t nBits = message.BitsSize();
    for (size_t i = 0; i < nBits; ++i) {
        ui32 bit = message.GetBits(i);
        if (bit >= (ui32)TLangMask::BeginIndex && bit < (ui32)TLangMask::EndIndex)
            mask.Set((ELanguage)bit);
    }
    return mask;
}

TString SerializeReadableLanguage(ELanguage lg) {
    return NameByLanguage(lg);
}

ELanguage DeserializeReadableLanguage(const TStringBuf& nm) {
    return LanguageByName(nm.data());
}

ELanguage DeserializeReadableLanguageStrict(const TStringBuf& nm) {
    ELanguage res = LanguageByNameStrict(nm.data());
    if (res == LANG_MAX)
        ythrow yexception() << "unknown language name: \"" << nm << "\"";
    return res;
}

TString SerializeReadableLangMask(const TLangMask& mask) {
    TString res;
    for (ELanguage bit : mask) {
        if (!!res)
            res += ',';
        res += SerializeReadableLanguage(bit);
    }
    if (!res)
        res = SerializeReadableLanguage(LANG_UNK);
    return res;
}

template <typename TLangNameGetter>
static TLangMask DeserializeReadableLangMaskImpl(const TString& nm, const TLangNameGetter& lng) {
    TLangMask mask;
    TVector<TString> names = SplitString(nm.c_str(), ",");
    for (const auto& name : names)
        mask.SafeSet(lng(name));
    return mask;
}

TLangMask DeserializeReadableLangMask(const TString& nm) {
    return DeserializeReadableLangMaskImpl(nm, DeserializeReadableLanguage);
}

TLangMask DeserializeReadableLangMaskStrict(const TString& nm) {
    return DeserializeReadableLangMaskImpl(nm, DeserializeReadableLanguageStrict);
}
