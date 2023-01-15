#pragma once

#include <library/langmask/langmask.h>

namespace NProto {
    class TLangMask;
}

void Serialize(NProto::TLangMask& message, const TLangMask& mask, bool humanReadable);
TLangMask Deserialize(const NProto::TLangMask& message);

TString SerializeReadableLanguage(ELanguage lg);
ELanguage DeserializeReadableLanguage(const TStringBuf& nm);
ELanguage DeserializeReadableLanguageStrict(const TStringBuf& nm);

TString SerializeReadableLangMask(const TLangMask& mask);
TLangMask DeserializeReadableLangMask(const TString& nm);
TLangMask DeserializeReadableLangMaskStrict(const TString& nm);
