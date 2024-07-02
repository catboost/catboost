#pragma once

#include "format_arg.h"

#include <library/cpp/yt/misc/enum.h>

#include <optional>

namespace NYT {

////////////////////////////////////////////////////////////////////////////////

std::optional<TString> TryDecodeEnumValue(TStringBuf value);
TString DecodeEnumValue(TStringBuf value);
TString EncodeEnumValue(TStringBuf value);

template <class T>
std::optional<T> TryParseEnum(TStringBuf value);

template <class T>
T ParseEnum(TStringBuf value);

template <class T>
void FormatEnum(TStringBuilderBase* builder, T value, bool lowerCase);

template <class T>
TString FormatEnum(T value);

////////////////////////////////////////////////////////////////////////////////

} // namespace NYT

#define ENUM_INL_H_
#include "enum-inl.h"
#undef ENUM_INL_H_
