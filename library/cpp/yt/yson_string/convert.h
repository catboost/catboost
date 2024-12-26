#pragma once

#include "string.h"

#include <library/cpp/yt/misc/guid.h>

#include <library/cpp/yt/exception/exception.h>

#include <util/generic/string.h>

#include <util/datetime/base.h>

namespace NYT::NYson {

////////////////////////////////////////////////////////////////////////////////

// Generic forward declarations.

template <class T>
TYsonString ConvertToYsonString(const T& value);

template <class T>
TYsonString ConvertToYsonString(const T& value, EYsonFormat format);

template <class T>
T ConvertFromYsonString(const TYsonStringBuf& str);

////////////////////////////////////////////////////////////////////////////////
// Basic specializations for ConvertToYsonString.

template <>
TYsonString ConvertToYsonString<i8>(const i8& value);
template <>
TYsonString ConvertToYsonString<i32>(const i32& value);
template <>
TYsonString ConvertToYsonString<i64>(const i64& value);

template <>
TYsonString ConvertToYsonString<ui8>(const ui8& value);
template <>
TYsonString ConvertToYsonString<ui32>(const ui32& value);
template <>
TYsonString ConvertToYsonString<ui64>(const ui64& value);

template <>
TYsonString ConvertToYsonString<TString>(const TString& value);
template <>
TYsonString ConvertToYsonString<TStringBuf>(const TStringBuf& value);
TYsonString ConvertToYsonString(const char* value);

template <>
TYsonString ConvertToYsonString<float>(const float& value);
template <>
TYsonString ConvertToYsonString<double>(const double& value);

template <>
TYsonString ConvertToYsonString<bool>(const bool& value);

template <>
TYsonString ConvertToYsonString<TInstant>(const TInstant& value);

template <>
TYsonString ConvertToYsonString<TDuration>(const TDuration& value);

template <>
TYsonString ConvertToYsonString<TGuid>(const TGuid& value);

////////////////////////////////////////////////////////////////////////////////
// Basic specializations for ConvertFromYsonString.
// Note: these currently support a subset of NYT::NYTree::Convert features.

class TYsonLiteralParseException
    : public TSimpleException
{
public:
    using TSimpleException::TSimpleException;
};

template <>
i8 ConvertFromYsonString<i8>(const TYsonStringBuf& str);
template <>
i32 ConvertFromYsonString<i32>(const TYsonStringBuf& str);
template <>
i64 ConvertFromYsonString<i64>(const TYsonStringBuf& str);

template <>
ui8 ConvertFromYsonString<ui8>(const TYsonStringBuf& str);
template <>
ui32 ConvertFromYsonString<ui32>(const TYsonStringBuf& str);
template <>
ui64 ConvertFromYsonString<ui64>(const TYsonStringBuf& str);

template <>
TString ConvertFromYsonString<TString>(const TYsonStringBuf& str);
template <>
std::string ConvertFromYsonString<std::string>(const TYsonStringBuf& str);

template <>
float ConvertFromYsonString<float>(const TYsonStringBuf& str);
template <>
double ConvertFromYsonString<double>(const TYsonStringBuf& str);

template <>
bool ConvertFromYsonString<bool>(const TYsonStringBuf& str);

template <>
TInstant ConvertFromYsonString<TInstant>(const TYsonStringBuf& str);

template <>
TDuration ConvertFromYsonString<TDuration>(const TYsonStringBuf& str);

template <>
TGuid ConvertFromYsonString<TGuid>(const TYsonStringBuf& str);

////////////////////////////////////////////////////////////////////////////////

} // namespace NYT::NYson
