/*
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <sstream>
#include <util/generic/string.h>

namespace ONNX_NAMESPACE {

#if defined(__ANDROID__)
template <typename T>
TString to_string(T value) {
  std::ostringstream os;
  os << value;
  return os.str();
}

inline int stoi(const TString& str) {
  std::stringstream ss;
  int n = 0;
  ss << str;
  ss >> n;
  return n;
}

#else
using std::stoi;
using std::to_string;
#endif // defined(__ANDROID__)

inline void MakeStringInternal(std::stringstream& /*ss*/) {}

template <typename T>
inline void MakeStringInternal(std::stringstream& ss, const T& t) {
  ss << t;
}

template <typename T, typename... Args>
inline void MakeStringInternal(std::stringstream& ss, const T& t, const Args&... args) {
  MakeStringInternal(ss, t);
  MakeStringInternal(ss, args...);
}

template <typename... Args>
TString MakeString(const Args&... args) {
  std::stringstream ss;
  MakeStringInternal(ss, args...);
  return TString(ss.str());
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
