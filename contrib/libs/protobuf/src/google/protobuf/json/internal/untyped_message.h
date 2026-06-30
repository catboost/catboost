#include "y_absl/log/absl_check.h"
// Protocol Buffers - Google's data interchange format
// Copyright 2008 Google Inc.  All rights reserved.
// https://developers.google.com/protocol-buffers/
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
//     * Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//     * Redistributions in binary form must reproduce the above
// copyright notice, this list of conditions and the following disclaimer
// in the documentation and/or other materials provided with the
// distribution.
//     * Neither the name of Google Inc. nor the names of its
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
// A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
// OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
// LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
// DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
// THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#ifndef GOOGLE_PROTOBUF_UITL_UNTYPED_MESSAGE_H__
#define GOOGLE_PROTOBUF_UITL_UNTYPED_MESSAGE_H__

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "google/protobuf/type.pb.h"
#include "google/protobuf/descriptor.h"
#include "google/protobuf/dynamic_message.h"
#include "google/protobuf/message.h"
#include "y_absl/container/flat_hash_map.h"
#include "y_absl/status/status.h"
#include "y_absl/strings/str_format.h"
#include "y_absl/strings/string_view.h"
#include "y_absl/types/optional.h"
#include "y_absl/types/span.h"
#include "y_absl/types/variant.h"
#include "google/protobuf/io/coded_stream.h"
#include "google/protobuf/util/type_resolver.h"
#include "google/protobuf/wire_format.h"
#include "google/protobuf/wire_format_lite.h"
#include "google/protobuf/stubs/status_macros.h"

// Must be included last.
#include "google/protobuf/port_def.inc"

namespace google {
namespace protobuf {
namespace json_internal {
struct SizeVisitor {
  template <typename T>
  size_t operator()(const std::vector<T>& x) {
    return x.size();
  }

  template <typename T>
  size_t operator()(const T& x) {
    return 1;
  }
};

// A DescriptorPool-like type for caching lookups from a TypeResolver.
//
// This type and all of its nested types are thread-hostile.
class ResolverPool {
 public:
  class Message;
  class Enum;
  class Field {
   public:
    Field(const Field&) = delete;
    Field& operator=(const Field&) = delete;

    y_absl::StatusOr<const Message*> MessageType() const;
    y_absl::StatusOr<const Enum*> EnumType() const;

    const Message& parent() const { return *parent_; }
    const google::protobuf::Field& proto() const { return *raw_; }

   private:
    friend class ResolverPool;

    Field() = default;

    ResolverPool* pool_ = nullptr;
    const google::protobuf::Field* raw_ = nullptr;
    const Message* parent_ = nullptr;
    mutable const void* type_ = nullptr;
  };

  class Message {
   public:
    Message(const Message&) = delete;
    Message& operator=(const Message&) = delete;

    y_absl::Span<const Field> FieldsByIndex() const;
    const Field* FindField(y_absl::string_view name) const;
    const Field* FindField(arc_i32 number) const;

    const google::protobuf::Type& proto() const { return raw_; }
    ResolverPool* pool() const { return pool_; }

   private:
    friend class ResolverPool;

    explicit Message(ResolverPool* pool) : pool_(pool) {}

    ResolverPool* pool_;
    google::protobuf::Type raw_;
    mutable std::unique_ptr<Field[]> fields_;
    mutable y_absl::flat_hash_map<y_absl::string_view, const Field*>
        fields_by_name_;
    mutable y_absl::flat_hash_map<arc_i32, const Field*> fields_by_number_;
  };

  class Enum {
   public:
    Enum(const Enum&) = delete;
    Enum& operator=(const Enum&) = delete;

    const google::protobuf::Enum& proto() const { return raw_; }
    ResolverPool* pool() const { return pool_; }

   private:
    friend class ResolverPool;

    explicit Enum(ResolverPool* pool) : pool_(pool) {}

    ResolverPool* pool_;
    google::protobuf::Enum raw_;
    mutable y_absl::flat_hash_map<y_absl::string_view, google::protobuf::EnumValue*>
        values_;
  };

  explicit ResolverPool(google::protobuf::util::TypeResolver* resolver)
      : resolver_(resolver) {}

  ResolverPool(const ResolverPool&) = delete;
  ResolverPool& operator=(const ResolverPool&) = delete;

  y_absl::StatusOr<const Message*> FindMessage(y_absl::string_view url);
  y_absl::StatusOr<const Enum*> FindEnum(y_absl::string_view url);

 private:
  y_absl::flat_hash_map<TProtoStringType, std::unique_ptr<Message>> messages_;
  y_absl::flat_hash_map<TProtoStringType, std::unique_ptr<Enum>> enums_;
  google::protobuf::util::TypeResolver* resolver_;
};

// A parsed wire-format proto that uses TypeReslover for parsing.
//
// This type is an implementation detail of the JSON parser.
class UntypedMessage final {
 public:
  // New nominal type instead of `bool` to avoid vector<bool> shenanigans.
  enum Bool : unsigned char { kTrue, kFalse };
  using Value = y_absl::variant<Bool, arc_i32, arc_ui32, arc_i64, arc_ui64, float,
                              double, TProtoStringType, UntypedMessage,
                              //
                              std::vector<Bool>, std::vector<arc_i32>,
                              std::vector<arc_ui32>, std::vector<arc_i64>,
                              std::vector<arc_ui64>, std::vector<float>,
                              std::vector<double>, std::vector<TProtoStringType>,
                              std::vector<UntypedMessage>>;

  UntypedMessage(const UntypedMessage&) = delete;
  UntypedMessage& operator=(const UntypedMessage&) = delete;
  UntypedMessage(UntypedMessage&&) = default;
  UntypedMessage& operator=(UntypedMessage&&) = default;

  // Tries to parse a proto with the given descriptor from an input stream.
  static y_absl::StatusOr<UntypedMessage> ParseFromStream(
      const ResolverPool::Message* desc, io::CodedInputStream& stream) {
    UntypedMessage msg(std::move(desc));
    RETURN_IF_ERROR(msg.Decode(stream));
    return std::move(msg);
  }

  // Returns the number of elements in a field by number.
  //
  // Optional fields are treated like repeated fields with one or zero elements.
  size_t Count(arc_i32 field_number) const {
    auto it = fields_.find(field_number);
    if (it == fields_.end()) {
      return 0;
    }

    return y_absl::visit(SizeVisitor{}, it->second);
  }

  // Returns the contents of a field by number.
  //
  // Optional fields are treated like repeated fields with one or zero elements.
  // If the field is not set, returns an empty span.
  //
  // If `T` is the wrong type, this function crashes.
  template <typename T>
  y_absl::Span<const T> Get(arc_i32 field_number) const {
    auto it = fields_.find(field_number);
    if (it == fields_.end()) {
      return {};
    }

    if (auto* val = y_absl::get_if<T>(&it->second)) {
      return y_absl::Span<const T>(val, 1);
    } else if (auto* vec = y_absl::get_if<std::vector<T>>(&it->second)) {
      return *vec;
    } else {
      Y_ABSL_CHECK(false) << "wrong type for UntypedMessage::Get(" << field_number
                        << ")";
      return {};  // avoid compiler warning.
    }
  }

  const ResolverPool::Message& desc() const { return *desc_; }

 private:
  enum Cardinality { kSingular, kRepeated };

  explicit UntypedMessage(const ResolverPool::Message* desc) : desc_(desc) {}

  y_absl::Status Decode(io::CodedInputStream& stream,
                      y_absl::optional<arc_i32> current_group = y_absl::nullopt);

  y_absl::Status DecodeVarint(io::CodedInputStream& stream,
                            const ResolverPool::Field& field);
  y_absl::Status Decode64Bit(io::CodedInputStream& stream,
                           const ResolverPool::Field& field);
  y_absl::Status Decode32Bit(io::CodedInputStream& stream,
                           const ResolverPool::Field& field);
  y_absl::Status DecodeDelimited(io::CodedInputStream& stream,
                               const ResolverPool::Field& field);

  template <typename T>
  y_absl::Status InsertField(const ResolverPool::Field& field, T value);

  const ResolverPool::Message* desc_;
  y_absl::flat_hash_map<arc_i32, Value> fields_;
};
}  // namespace json_internal
}  // namespace protobuf
}  // namespace google

#include "google/protobuf/port_undef.inc"
#endif  // GOOGLE_PROTOBUF_UITL_UNTYPED_MESSAGE_H__
