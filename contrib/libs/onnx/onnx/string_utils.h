#pragma once

#include <util/generic/string.h>
#include <util/stream/str.h>
#include <util/string/cast.h>

namespace ONNX_NAMESPACE {


template <typename T>
TString to_string(T value) {
  return ToString(value);
}

inline int stoi(const TString& str) {
  return FromString<int>(str);
}


inline void MakeStringInternal(TStringStream& /*ss*/) {}

template <typename T>
inline void MakeStringInternal(TStringStream& ss, const T& t) {
  ss << t;
}

template <typename T, typename... Args>
inline void
MakeStringInternal(TStringStream& ss, const T& t, const Args&... args) {
  MakeStringInternal(ss, t);
  MakeStringInternal(ss, args...);
}

template <typename... Args>
TString MakeString(const Args&... args) {
  TStringStream ss;
  MakeStringInternal(ss, args...);
  return TString(ss.Str());
}

// Specializations for already-a-string types.
template <>
inline TString MakeString(const TString& str) {
  return str;
}
inline TString MakeString(const char* c_str) {
  return TString(c_str);
}
} // namespace ONNX_NAMESPACE
