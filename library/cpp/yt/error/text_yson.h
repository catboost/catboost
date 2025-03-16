#pragma once

#include <library/cpp/yt/misc/guid.h>

#include <util/datetime/base.h>

#include <util/generic/strbuf.h>

#include <string>

namespace NYT::NDetail {

////////////////////////////////////////////////////////////////////////////////

size_t FloatToStringWithNanInf(double value, char* buf, size_t size);

////////////////////////////////////////////////////////////////////////////////

bool IsBinaryYson(TStringBuf str);

////////////////////////////////////////////////////////////////////////////////

template <class T>
std::string ConvertToTextYsonString(const T& value) = delete;
template <class T>
T ConvertFromTextYsonString(TStringBuf str) = delete;

////////////////////////////////////////////////////////////////////////////////

template <>
std::string ConvertToTextYsonString<i8>(const i8& value);
template <>
std::string ConvertToTextYsonString<i32>(const i32& value);
template <>
std::string ConvertToTextYsonString<i64>(const i64& value);

template <>
std::string ConvertToTextYsonString<ui8>(const ui8& value);
template <>
std::string ConvertToTextYsonString<ui32>(const ui32& value);
template <>
std::string ConvertToTextYsonString<ui64>(const ui64& value);

template <>
std::string ConvertToTextYsonString<TStringBuf>(const TStringBuf& value);

template <>
std::string ConvertToTextYsonString<float>(const float& value);
template <>
std::string ConvertToTextYsonString<double>(const double& value);

template <>
std::string ConvertToTextYsonString<bool>(const bool& value);

template <>
std::string ConvertToTextYsonString<TInstant>(const TInstant& value);

template <>
std::string ConvertToTextYsonString<TDuration>(const TDuration& value);

template <>
std::string ConvertToTextYsonString<TGuid>(const TGuid& value);

////////////////////////////////////////////////////////////////////////////////

template <>
i8 ConvertFromTextYsonString<i8>(TStringBuf str);
template <>
i32 ConvertFromTextYsonString<i32>(TStringBuf str);
template <>
i64 ConvertFromTextYsonString<i64>(TStringBuf str);

template <>
ui8 ConvertFromTextYsonString<ui8>(TStringBuf str);
template <>
ui32 ConvertFromTextYsonString<ui32>(TStringBuf str);
template <>
ui64 ConvertFromTextYsonString<ui64>(TStringBuf str);

template <>
TString ConvertFromTextYsonString<TString>(TStringBuf str);
template <>
std::string ConvertFromTextYsonString<std::string>(TStringBuf str);

template <>
float ConvertFromTextYsonString<float>(TStringBuf str);
template <>
double ConvertFromTextYsonString<double>(TStringBuf str);

template <>
bool ConvertFromTextYsonString<bool>(TStringBuf str);

template <>
TInstant ConvertFromTextYsonString<TInstant>(TStringBuf str);

template <>
TDuration ConvertFromTextYsonString<TDuration>(TStringBuf str);

template <>
TGuid ConvertFromTextYsonString<TGuid>(TStringBuf str);

////////////////////////////////////////////////////////////////////////////////

} // namespace NYT::NDetail
