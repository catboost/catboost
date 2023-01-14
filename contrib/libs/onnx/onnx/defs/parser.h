/*
 * SPDX-License-Identifier: Apache-2.0
 */

// Experimental language syntax and parser for ONNX. Please note that the syntax as formalized
// by this parser is preliminary and may change.

#pragma once

#include <ctype.h>
#include <iostream>
#include <stdexcept>
#include <util/generic/string.h>
#include <unordered_map>

#include "onnx/onnx_pb.h"

#include "onnx/common/status.h"
#include "onnx/string_utils.h"

namespace ONNX_NAMESPACE {

using namespace ONNX_NAMESPACE::Common;

using IdList = google::protobuf::RepeatedPtrField<TString>;

using NodeList = google::protobuf::RepeatedPtrField<NodeProto>;

using AttrList = google::protobuf::RepeatedPtrField<AttributeProto>;

using ValueInfoList = google::protobuf::RepeatedPtrField<ValueInfoProto>;

using TensorList = google::protobuf::RepeatedPtrField<TensorProto>;

using OpsetIdList = google::protobuf::RepeatedPtrField<OperatorSetIdProto>;

#define CHECK_PARSER_STATUS(status) \
  {                                 \
    auto local_status_ = status;    \
    if (!local_status_.IsOK())      \
      return local_status_;         \
  }

template <typename Map>
class StringIntMap {
 public:
  static const std::unordered_map<TString, int32_t>& Instance() {
    static Map instance;
    return instance.map_;
  }

  static int32_t Lookup(const TString& dtype) {
    auto it = Instance().find(dtype);
    if (it != Instance().end())
      return it->second;
    return 0;
  }

  static const TString& ToString(int32_t dtype) {
    static TString undefined("undefined");
    for (const auto& pair : Instance()) {
      if (pair.second == dtype)
        return pair.first;
    }
    return undefined;
  }

 protected:
  std::unordered_map<TString, int32_t> map_;
};

class PrimitiveTypeNameMap : public StringIntMap<PrimitiveTypeNameMap> {
 public:
  PrimitiveTypeNameMap() : StringIntMap() {
    map_["float"] = 1;
    map_["uint8"] = 2;
    map_["int8"] = 3;
    map_["uint16"] = 4;
    map_["int16"] = 5;
    map_["int32"] = 6;
    map_["int64"] = 7;
    map_["string"] = 8;
    map_["bool"] = 9;
    map_["float16"] = 10;
    map_["double"] = 11;
    map_["uint32"] = 12;
    map_["uint64"] = 13;
    map_["complex64"] = 14;
    map_["complex128"] = 15;
    map_["bfloat16"] = 16;
  }

  static bool IsTypeName(const TString& dtype) {
    return Lookup(dtype) != 0;
  }
};

class AttributeTypeNameMap : public StringIntMap<AttributeTypeNameMap> {
 public:
  AttributeTypeNameMap() : StringIntMap() {
    map_["float"] = 1;
    map_["int"] = 2;
    map_["string"] = 3;
    map_["tensor"] = 4;
    map_["graph"] = 5;
    map_["sparse_tensor"] = 11;
    map_["type_proto"] = 13;
    map_["floats"] = 6;
    map_["ints"] = 7;
    map_["strings"] = 8;
    map_["tensors"] = 9;
    map_["graphs"] = 10;
    map_["sparse_tensors"] = 12;
    map_["type_protos"] = 14;
  }
};

class KeyWordMap {
 public:
  enum class KeyWord {
    NONE,
    IR_VERSION,
    OPSET_IMPORT,
    PRODUCER_NAME,
    PRODUCER_VERSION,
    DOMAIN_KW,
    MODEL_VERSION,
    DOC_STRING,
    METADATA_PROPS,
    SEQ_TYPE,
    MAP_TYPE,
    OPTIONAL_TYPE,
    SPARSE_TENSOR_TYPE
  };

  KeyWordMap() {
    map_["ir_version"] = KeyWord::IR_VERSION;
    map_["opset_import"] = KeyWord::OPSET_IMPORT;
    map_["producer_name"] = KeyWord::PRODUCER_NAME;
    map_["producer_version"] = KeyWord::PRODUCER_VERSION;
    map_["domain"] = KeyWord::DOMAIN_KW;
    map_["model_version"] = KeyWord::MODEL_VERSION;
    map_["doc_string"] = KeyWord::DOC_STRING;
    map_["metadata_props"] = KeyWord::METADATA_PROPS;
    map_["seq"] = KeyWord::SEQ_TYPE;
    map_["map"] = KeyWord::MAP_TYPE;
    map_["optional"] = KeyWord::OPTIONAL_TYPE;
    map_["sparse_tensor"] = KeyWord::SPARSE_TENSOR_TYPE;
  }

  static const std::unordered_map<TString, KeyWord>& Instance() {
    static KeyWordMap instance;
    return instance.map_;
  }

  static KeyWord Lookup(const TString& id) {
    auto it = Instance().find(id);
    if (it != Instance().end())
      return it->second;
    return KeyWord::NONE;
  }

  static const TString& ToString(KeyWord kw) {
    static TString undefined("undefined");
    for (const auto& pair : Instance()) {
      if (pair.second == kw)
        return pair.first;
    }
    return undefined;
  }

 private:
  std::unordered_map<TString, KeyWord> map_;
};

class ParserBase {
 public:
  ParserBase(const TString& str)
      : start_(str.data()), next_(str.data()), end_(str.data() + str.length()), saved_pos_(next_) {}

  ParserBase(const char* cstr) : start_(cstr), next_(cstr), end_(cstr + strlen(cstr)), saved_pos_(next_) {}

  void SavePos() {
    saved_pos_ = next_;
  }

  void RestorePos() {
    next_ = saved_pos_;
  }

  TString GetCurrentPos() {
    uint32_t line = 1, col = 1;
    for (const char* p = start_; p < next_; ++p) {
      if (*p == '\n') {
        ++line;
        col = 1;
      } else {
        ++col;
      }
    }
    return ONNX_NAMESPACE::MakeString("(line: ", line, " column: ", col, ")");
  }

  // Return a suitable suffix of what has been parsed to provide error message context:
  // return the line containing the last non-space character preceding the error (if it exists).
  TString GetErrorContext() {
    // Special cases: empty input string, and parse-error at first character.
    const char* p = next_ < end_ ? next_ : next_ - 1;
    while ((p > start_) && isspace(*p))
      --p;
    while ((p > start_) && (*p != '\n'))
      --p;
    // Start at character after '\n' unless we are at start of input
    const char* context_start = (p > start_) ? (p + 1) : start_;
    for (p = context_start; (p < end_) && (*p != '\n'); ++p)
      ;
    return TString(context_start, p - context_start);
  }

  template <typename... Args>
  Status ParseError(const Args&... args) {
    return Status(
        NONE,
        FAIL,
        ONNX_NAMESPACE::MakeString(
            "[ParseError at position ", GetCurrentPos(), "]\n", "Error context: ", GetErrorContext(), "\n", args...));
  }

  void SkipWhiteSpace() {
    do {
      while ((next_ < end_) && (isspace(*next_)))
        ++next_;
      if ((next_ >= end_) || ((*next_) != '#'))
        return;
      // Skip rest of the line:
      while ((next_ < end_) && ((*next_) != '\n'))
        ++next_;
    } while (true);
  }

  int NextChar(bool skipspace = true) {
    if (skipspace)
      SkipWhiteSpace();
    return (next_ < end_) ? *next_ : 0;
  }

  bool Matches(char ch, bool skipspace = true) {
    if (skipspace)
      SkipWhiteSpace();
    if ((next_ < end_) && (*next_ == ch)) {
      ++next_;
      return true;
    }
    return false;
  }

  Status Match(char ch, bool skipspace = true) {
    if (!Matches(ch, skipspace))
      return ParseError("Expected character ", ch, " not found.");
    return Status::OK();
  }

  bool EndOfInput() {
    SkipWhiteSpace();
    return (next_ >= end_);
  }

  enum class LiteralType { INT_LITERAL, FLOAT_LITERAL, STRING_LITERAL };

  struct Literal {
    LiteralType type;
    TString value;
  };

  Status Parse(Literal& result);

  Status Parse(int64_t& val) {
    Literal literal;
    CHECK_PARSER_STATUS(Parse(literal));
    if (literal.type != LiteralType::INT_LITERAL)
      return ParseError("Integer value expected, but not found.");
    TString s = literal.value;
    val = std::stoll(s);
    return Status::OK();
  }

  Status Parse(uint64_t& val) {
    Literal literal;
    CHECK_PARSER_STATUS(Parse(literal));
    if (literal.type != LiteralType::INT_LITERAL)
      return ParseError("Integer value expected, but not found.");
    TString s = literal.value;
    val = std::stoull(s);
    return Status::OK();
  }

  Status Parse(float& val) {
    Literal literal;
    CHECK_PARSER_STATUS(Parse(literal));
    switch (literal.type) {
      case LiteralType::INT_LITERAL:
      case LiteralType::FLOAT_LITERAL:
        val = std::stof(literal.value);
        break;
      default:
        return ParseError("Unexpected literal type.");
    }
    return Status::OK();
  }

  Status Parse(double& val) {
    Literal literal;
    CHECK_PARSER_STATUS(Parse(literal));
    switch (literal.type) {
      case LiteralType::INT_LITERAL:
      case LiteralType::FLOAT_LITERAL:
        val = std::stod(literal.value);
        break;
      default:
        return ParseError("Unexpected literal type.");
    }
    return Status::OK();
  }

  // Parse a string-literal enclosed within doube-quotes.
  Status Parse(TString& val) {
    Literal literal;
    CHECK_PARSER_STATUS(Parse(literal));
    if (literal.type != LiteralType::STRING_LITERAL)
      return ParseError("String value expected, but not found.");
    val = literal.value;
    return Status::OK();
  }

  // Parse an identifier, including keywords. If none found, this will
  // return an empty-string identifier.
  Status ParseOptionalIdentifier(TString& id) {
    SkipWhiteSpace();
    auto from = next_;
    if ((next_ < end_) && (isalpha(*next_) || (*next_ == '_'))) {
      ++next_;
      while ((next_ < end_) && (isalnum(*next_) || (*next_ == '_')))
        ++next_;
    }
    id = TString(from, next_ - from);
    return Status::OK();
  }

  Status ParseIdentifier(TString& id) {
    ParseOptionalIdentifier(id);
    if (id.empty())
      return ParseError("Identifier expected but not found.");
    return Status::OK();
  }

  Status PeekIdentifier(TString& id) {
    SavePos();
    ParseOptionalIdentifier(id);
    RestorePos();
    return Status::OK();
  }

  Status Parse(KeyWordMap::KeyWord& keyword) {
    TString id;
    CHECK_PARSER_STATUS(ParseIdentifier(id));
    keyword = KeyWordMap::Lookup(id);
    return Status::OK();
  }

 protected:
  const char* start_;
  const char* next_;
  const char* end_;
  const char* saved_pos_;
};

class OnnxParser : public ParserBase {
 public:
  OnnxParser(const char* cstr) : ParserBase(cstr) {}

  Status Parse(TensorShapeProto& shape);

  Status Parse(TypeProto& typeProto);

  Status Parse(TensorProto& tensorProto);

  Status Parse(AttributeProto& attr);

  Status Parse(AttrList& attrlist);

  Status Parse(NodeProto& node);

  Status Parse(NodeList& nodelist);

  Status Parse(GraphProto& graph);

  Status Parse(FunctionProto& fn);

  Status Parse(ModelProto& model);

  template <typename T>
  static Status Parse(T& parsedData, const char* input) {
    OnnxParser parser(input);
    return parser.Parse(parsedData);
  }

 private:
  Status Parse(TString name, GraphProto& graph);

  Status Parse(IdList& idlist);

  Status Parse(char open, IdList& idlist, char close);

  Status ParseSingleAttributeValue(AttributeProto& attr);

  Status Parse(ValueInfoProto& valueinfo);

  Status Parse(ValueInfoList& vilist);

  Status ParseInput(ValueInfoList& vilist, TensorList& initializers);

  Status ParseValueInfo(ValueInfoList& vilist, TensorList& initializers);

  Status Parse(TensorProto& tensorProto, const TypeProto& tensorTypeProto);

  Status Parse(OpsetIdList& opsets);

  bool NextIsType();
};

} // namespace ONNX_NAMESPACE