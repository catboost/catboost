/*
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef ONNX_DATA_TYPE_UTILS_H
#define ONNX_DATA_TYPE_UTILS_H

#include <mutex>
#include <util/generic/string.h>
#include <unordered_map>
#include <unordered_set>
#include "onnx/common/common.h"
#include "onnx/onnx_pb.h"

namespace ONNX_NAMESPACE {
// String pointer as unique TypeProto identifier.
using DataType = const TString*;

namespace Utils {

// Data type utility, which maintains a global type string to TypeProto map.
// DataType (string pointer) is used as unique data type identifier for
// efficiency.
//
// Grammar for data type string:
// <type> ::= <data_type> |
//            tensor(<data_type>) |
//            seq(<type>) |
//            map(<data_type>, <type>)
// <data_type> :: = float | int32 | string | bool | uint8
//                | int8 | uint16 | int16 | int64 | float16 | double
//
// NOTE: <type> ::= <data_type> means the data is scalar (zero dimension).
//
// Example: float, tensor(float), etc.
//
class DataTypeUtils final {
 public:
  // If the DataType input is invalid, this function will throw std::invalid_argument exception.
  // If ONNX_NO_EXCEPTIONS is set it will abort.
  static DataType ToType(const TString& type_str);

  // If the DataType input is invalid, this function will throw std::invalid_argument exception.
  // If ONNX_NO_EXCEPTIONS is set it will abort.
  static DataType ToType(const TypeProto& type_proto);

  // If the DataType input is invalid, this function will throw std::invalid_argument exception.
  // If ONNX_NO_EXCEPTIONS is set it will abort.
  static const TypeProto& ToTypeProto(const DataType& data_type);
  static TString ToDataTypeString(int32_t tensor_data_type);

 private:
  static void FromString(const TString& type_str, TypeProto& type_proto);

  static void FromDataTypeString(const TString& type_str, int32_t& tensor_data_type);

  static TString ToString(const TypeProto& type_proto, const TString& left = "", const TString& right = "");

  // If int32_t input is invalid, this function will throw an exception.
  // If ONNX_NO_EXCEPTIONS is set it will abort.

  static bool IsValidDataTypeString(const TString& type_str);

  static std::unordered_map<TString, TypeProto>& GetTypeStrToProtoMap();

  // Returns lock used for concurrent updates to TypeStrToProtoMap.
  static std::mutex& GetTypeStrLock();
};
} // namespace Utils
} // namespace ONNX_NAMESPACE

#endif // ! ONNX_DATA_TYPE_UTILS_H
