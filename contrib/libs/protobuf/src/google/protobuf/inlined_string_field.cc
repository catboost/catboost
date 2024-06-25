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

#include <google/protobuf/inlined_string_field.h>

#include <google/protobuf/arena.h>
#include <google/protobuf/arenastring.h>
#include <google/protobuf/generated_message_util.h>
#include <google/protobuf/message_lite.h>
#include <google/protobuf/parse_context.h>

// clang-format off
#include <google/protobuf/port_def.inc>
// clang-format on

namespace google {
namespace protobuf {
namespace internal {


TProtoStringType* InlinedStringField::Mutable(const LazyString& /*default_value*/,
                                         Arena* arena, bool donated,
                                         arc_ui32* donating_states,
                                         arc_ui32 mask, MessageLite* msg) {
  if (arena == nullptr || !donated) {
    return UnsafeMutablePointer();
  }
  return MutableSlow(arena, donated, donating_states, mask, msg);
}

TProtoStringType* InlinedStringField::Mutable(Arena* arena, bool donated,
                                         arc_ui32* donating_states,
                                         arc_ui32 mask, MessageLite* msg) {
  if (arena == nullptr || !donated) {
    return UnsafeMutablePointer();
  }
  return MutableSlow(arena, donated, donating_states, mask, msg);
}

TProtoStringType* InlinedStringField::MutableSlow(::google::protobuf::Arena* arena,
                                             bool donated,
                                             arc_ui32* donating_states,
                                             arc_ui32 mask, MessageLite* msg) {
  (void)mask;
  (void)msg;
  return UnsafeMutablePointer();
}

void InlinedStringField::SetAllocated(const TProtoStringType* default_value,
                                      TProtoStringType* value, Arena* arena,
                                      bool donated, arc_ui32* donating_states,
                                      arc_ui32 mask, MessageLite* msg) {
  (void)mask;
  (void)msg;
  SetAllocatedNoArena(default_value, value);
}

void InlinedStringField::Set(TProtoStringType&& value, Arena* arena, bool donated,
                             arc_ui32* donating_states, arc_ui32 mask,
                             MessageLite* msg) {
  (void)donating_states;
  (void)mask;
  (void)msg;
  SetNoArena(std::move(value));
}

TProtoStringType* InlinedStringField::Release() {
  auto* released = new TProtoStringType(std::move(*get_mutable()));
  get_mutable()->clear();
  return released;
}

TProtoStringType* InlinedStringField::Release(Arena* arena, bool donated) {
  // We can not steal donated arena strings.
  TProtoStringType* released = (arena != nullptr && donated)
                              ? new TProtoStringType(*get_mutable())
                              : new TProtoStringType(std::move(*get_mutable()));
  get_mutable()->clear();
  return released;
}

void InlinedStringField::ClearToDefault(const LazyString& default_value,
                                        Arena* arena, bool donated) {
  (void)arena;
  get_mutable()->assign(default_value.get());
}


}  // namespace internal
}  // namespace protobuf
}  // namespace google
