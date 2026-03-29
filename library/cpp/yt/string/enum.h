#pragma once

#include "format_arg.h"

#include <library/cpp/yt/misc/enum.h>

#include <optional>

namespace NYT {

////////////////////////////////////////////////////////////////////////////////

std::optional<std::string> TryDecodeEnumValue(TStringBuf value);
std::string DecodeEnumValue(TStringBuf value);
std::string EncodeEnumValue(TStringBuf value);

template <class T>
std::optional<T> TryParseEnum(TStringBuf str, bool enableUnknown = false);

//! Parses an enum value from a string.
//! Link against library/cpp/yt/string/enable_enum_suggestions_on_enum_parse_error
//! to include suggestions in the exception message on parse failure.
template <class T>
T ParseEnum(TStringBuf str);

template <class T>
void FormatEnum(TStringBuilderBase* builder, T value, bool lowerCase);

template <class T>
std::string FormatEnum(T value);

////////////////////////////////////////////////////////////////////////////////

} // namespace NYT

#define ENUM_INL_H_
#include "enum-inl.h"
#undef ENUM_INL_H_
