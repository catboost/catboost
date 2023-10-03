//
//
// Copyright 2021 the gRPC authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
//

#include <grpc/support/port_platform.h>

#include "src/core/lib/gprpp/status_helper.h"

#include <string.h>

#include <algorithm>
#include <utility>

#include "y_absl/strings/cord.h"
#include "y_absl/strings/escaping.h"
#include "y_absl/strings/match.h"
#include "y_absl/strings/numbers.h"
#include "y_absl/strings/str_cat.h"
#include "y_absl/strings/str_join.h"
#include "y_absl/time/clock.h"
#include "google/protobuf/any.upb.h"
#include "google/rpc/status.upb.h"
#include "upb/arena.h"
#include "upb/upb.h"
#include "upb/upb.hpp"

#include <grpc/support/log.h>

#include "src/core/lib/slice/percent_encoding.h"
#include "src/core/lib/slice/slice.h"

#include <util/string/cast.h>

namespace grpc_core {

namespace {

#define TYPE_URL_PREFIX "type.googleapis.com/grpc.status."
#define TYPE_INT_TAG "int."
#define TYPE_STR_TAG "str."
#define TYPE_TIME_TAG "time."
#define TYPE_CHILDREN_TAG "children"
#define TYPE_URL(name) (TYPE_URL_PREFIX name)
const y_absl::string_view kTypeUrlPrefix = TYPE_URL_PREFIX;
const y_absl::string_view kTypeIntTag = TYPE_INT_TAG;
const y_absl::string_view kTypeStrTag = TYPE_STR_TAG;
const y_absl::string_view kTypeTimeTag = TYPE_TIME_TAG;
const y_absl::string_view kTypeChildrenTag = TYPE_CHILDREN_TAG;
const y_absl::string_view kChildrenPropertyUrl = TYPE_URL(TYPE_CHILDREN_TAG);

const char* GetStatusIntPropertyUrl(StatusIntProperty key) {
  switch (key) {
    case StatusIntProperty::kErrorNo:
      return TYPE_URL(TYPE_INT_TAG "errno");
    case StatusIntProperty::kFileLine:
      return TYPE_URL(TYPE_INT_TAG "file_line");
    case StatusIntProperty::kStreamId:
      return TYPE_URL(TYPE_INT_TAG "stream_id");
    case StatusIntProperty::kRpcStatus:
      return TYPE_URL(TYPE_INT_TAG "grpc_status");
    case StatusIntProperty::kOffset:
      return TYPE_URL(TYPE_INT_TAG "offset");
    case StatusIntProperty::kIndex:
      return TYPE_URL(TYPE_INT_TAG "index");
    case StatusIntProperty::kSize:
      return TYPE_URL(TYPE_INT_TAG "size");
    case StatusIntProperty::kHttp2Error:
      return TYPE_URL(TYPE_INT_TAG "http2_error");
    case StatusIntProperty::kTsiCode:
      return TYPE_URL(TYPE_INT_TAG "tsi_code");
    case StatusIntProperty::kWsaError:
      return TYPE_URL(TYPE_INT_TAG "wsa_error");
    case StatusIntProperty::kFd:
      return TYPE_URL(TYPE_INT_TAG "fd");
    case StatusIntProperty::kHttpStatus:
      return TYPE_URL(TYPE_INT_TAG "http_status");
    case StatusIntProperty::kOccurredDuringWrite:
      return TYPE_URL(TYPE_INT_TAG "occurred_during_write");
    case StatusIntProperty::ChannelConnectivityState:
      return TYPE_URL(TYPE_INT_TAG "channel_connectivity_state");
    case StatusIntProperty::kLbPolicyDrop:
      return TYPE_URL(TYPE_INT_TAG "lb_policy_drop");
  }
  GPR_UNREACHABLE_CODE(return "unknown");
}

const char* GetStatusStrPropertyUrl(StatusStrProperty key) {
  switch (key) {
    case StatusStrProperty::kDescription:
      return TYPE_URL(TYPE_STR_TAG "description");
    case StatusStrProperty::kFile:
      return TYPE_URL(TYPE_STR_TAG "file");
    case StatusStrProperty::kOsError:
      return TYPE_URL(TYPE_STR_TAG "os_error");
    case StatusStrProperty::kSyscall:
      return TYPE_URL(TYPE_STR_TAG "syscall");
    case StatusStrProperty::kTargetAddress:
      return TYPE_URL(TYPE_STR_TAG "target_address");
    case StatusStrProperty::kGrpcMessage:
      return TYPE_URL(TYPE_STR_TAG "grpc_message");
    case StatusStrProperty::kRawBytes:
      return TYPE_URL(TYPE_STR_TAG "raw_bytes");
    case StatusStrProperty::kTsiError:
      return TYPE_URL(TYPE_STR_TAG "tsi_error");
    case StatusStrProperty::kFilename:
      return TYPE_URL(TYPE_STR_TAG "filename");
    case StatusStrProperty::kKey:
      return TYPE_URL(TYPE_STR_TAG "key");
    case StatusStrProperty::kValue:
      return TYPE_URL(TYPE_STR_TAG "value");
  }
  GPR_UNREACHABLE_CODE(return "unknown");
}

const char* GetStatusTimePropertyUrl(StatusTimeProperty key) {
  switch (key) {
    case StatusTimeProperty::kCreated:
      return TYPE_URL(TYPE_TIME_TAG "created_time");
  }
  GPR_UNREACHABLE_CODE(return "unknown");
}

void EncodeUInt32ToBytes(uint32_t v, char* buf) {
  buf[0] = v & 0xFF;
  buf[1] = (v >> 8) & 0xFF;
  buf[2] = (v >> 16) & 0xFF;
  buf[3] = (v >> 24) & 0xFF;
}

uint32_t DecodeUInt32FromBytes(const char* buf) {
  const unsigned char* ubuf = reinterpret_cast<const unsigned char*>(buf);
  return ubuf[0] | (static_cast<uint32_t>(ubuf[1]) << 8) |
         (static_cast<uint32_t>(ubuf[2]) << 16) |
         (static_cast<uint32_t>(ubuf[3]) << 24);
}

std::vector<y_absl::Status> ParseChildren(y_absl::Cord children) {
  std::vector<y_absl::Status> result;
  upb::Arena arena;
  // Cord is flattened to iterate the buffer easily at the cost of memory copy.
  // TODO(veblush): Optimize this once CordReader is introduced.
  y_absl::string_view buf = children.Flatten();
  size_t cur = 0;
  while (buf.size() - cur >= sizeof(uint32_t)) {
    size_t msg_size = DecodeUInt32FromBytes(buf.data() + cur);
    cur += sizeof(uint32_t);
    GPR_ASSERT(buf.size() - cur >= msg_size);
    google_rpc_Status* msg =
        google_rpc_Status_parse(buf.data() + cur, msg_size, arena.ptr());
    cur += msg_size;
    result.push_back(internal::StatusFromProto(msg));
  }
  return result;
}

}  // namespace

y_absl::Status StatusCreate(y_absl::StatusCode code, y_absl::string_view msg,
                          const DebugLocation& location,
                          std::vector<y_absl::Status> children) {
  y_absl::Status s(code, msg);
  if (location.file() != nullptr) {
    StatusSetStr(&s, StatusStrProperty::kFile, location.file());
  }
  if (location.line() != -1) {
    StatusSetInt(&s, StatusIntProperty::kFileLine, location.line());
  }
  StatusSetTime(&s, StatusTimeProperty::kCreated, y_absl::Now());
  for (const y_absl::Status& child : children) {
    if (!child.ok()) {
      StatusAddChild(&s, child);
    }
  }
  return s;
}

void StatusSetInt(y_absl::Status* status, StatusIntProperty key, intptr_t value) {
  status->SetPayload(GetStatusIntPropertyUrl(key),
                     y_absl::Cord(::ToString(value)));
}

y_absl::optional<intptr_t> StatusGetInt(const y_absl::Status& status,
                                      StatusIntProperty key) {
  y_absl::optional<y_absl::Cord> p =
      status.GetPayload(GetStatusIntPropertyUrl(key));
  if (p.has_value()) {
    y_absl::optional<y_absl::string_view> sv = p->TryFlat();
    intptr_t value;
    if (sv.has_value()) {
      if (y_absl::SimpleAtoi(*sv, &value)) {
        return value;
      }
    } else {
      if (y_absl::SimpleAtoi(TString(*p), &value)) {
        return value;
      }
    }
  }
  return {};
}

void StatusSetStr(y_absl::Status* status, StatusStrProperty key,
                  y_absl::string_view value) {
  status->SetPayload(GetStatusStrPropertyUrl(key), y_absl::Cord(value));
}

y_absl::optional<TString> StatusGetStr(const y_absl::Status& status,
                                         StatusStrProperty key) {
  y_absl::optional<y_absl::Cord> p =
      status.GetPayload(GetStatusStrPropertyUrl(key));
  if (p.has_value()) {
    return TString(*p);
  }
  return {};
}

void StatusSetTime(y_absl::Status* status, StatusTimeProperty key,
                   y_absl::Time time) {
  TString time_str =
      y_absl::FormatTime(y_absl::RFC3339_full, time, y_absl::UTCTimeZone());
  status->SetPayload(GetStatusTimePropertyUrl(key),
                     y_absl::Cord(std::move(time_str)));
}

y_absl::optional<y_absl::Time> StatusGetTime(const y_absl::Status& status,
                                         StatusTimeProperty key) {
  y_absl::optional<y_absl::Cord> p =
      status.GetPayload(GetStatusTimePropertyUrl(key));
  if (p.has_value()) {
    y_absl::optional<y_absl::string_view> sv = p->TryFlat();
    y_absl::Time time;
    if (sv.has_value()) {
      if (y_absl::ParseTime(y_absl::RFC3339_full, sv.value(), &time, nullptr)) {
        return time;
      }
    } else {
      TString s = TString(*p);
      if (y_absl::ParseTime(y_absl::RFC3339_full, s, &time, nullptr)) {
        return time;
      }
    }
  }
  return {};
}

void StatusAddChild(y_absl::Status* status, y_absl::Status child) {
  upb::Arena arena;
  // Serialize msg to buf
  google_rpc_Status* msg = internal::StatusToProto(child, arena.ptr());
  size_t buf_len = 0;
  char* buf = google_rpc_Status_serialize(msg, arena.ptr(), &buf_len);
  // Append (msg-length and msg) to children payload
  y_absl::optional<y_absl::Cord> old_children =
      status->GetPayload(kChildrenPropertyUrl);
  y_absl::Cord children;
  if (old_children.has_value()) {
    children = *old_children;
  }
  char head_buf[sizeof(uint32_t)];
  EncodeUInt32ToBytes(buf_len, head_buf);
  children.Append(y_absl::string_view(head_buf, sizeof(uint32_t)));
  children.Append(y_absl::string_view(buf, buf_len));
  status->SetPayload(kChildrenPropertyUrl, std::move(children));
}

std::vector<y_absl::Status> StatusGetChildren(y_absl::Status status) {
  y_absl::optional<y_absl::Cord> children = status.GetPayload(kChildrenPropertyUrl);
  return children.has_value() ? ParseChildren(*children)
                              : std::vector<y_absl::Status>();
}

TString StatusToString(const y_absl::Status& status) {
  if (status.ok()) {
    return "OK";
  }
  TString head;
  y_absl::StrAppend(&head, y_absl::StatusCodeToString(status.code()));
  if (!status.message().empty()) {
    y_absl::StrAppend(&head, ":", status.message());
  }
  std::vector<TString> kvs;
  y_absl::optional<y_absl::Cord> children;
  status.ForEachPayload([&](y_absl::string_view type_url,
                            const y_absl::Cord& payload) {
    if (y_absl::StartsWith(type_url, kTypeUrlPrefix)) {
      type_url.remove_prefix(kTypeUrlPrefix.size());
      if (type_url == kTypeChildrenTag) {
        children = payload;
        return;
      }
      y_absl::string_view payload_view;
      TString payload_storage;
      if (payload.TryFlat().has_value()) {
        payload_view = payload.TryFlat().value();
      } else {
        payload_storage = TString(payload);
        payload_view = payload_storage;
      }
      if (y_absl::StartsWith(type_url, kTypeIntTag)) {
        type_url.remove_prefix(kTypeIntTag.size());
        kvs.push_back(y_absl::StrCat(type_url, ":", payload_view));
      } else if (y_absl::StartsWith(type_url, kTypeStrTag)) {
        type_url.remove_prefix(kTypeStrTag.size());
        kvs.push_back(y_absl::StrCat(type_url, ":\"",
                                   y_absl::CHexEscape(payload_view), "\""));
      } else if (y_absl::StartsWith(type_url, kTypeTimeTag)) {
        type_url.remove_prefix(kTypeTimeTag.size());
        y_absl::Time t;
        if (y_absl::ParseTime(y_absl::RFC3339_full, payload_view, &t, nullptr)) {
          kvs.push_back(
              y_absl::StrCat(type_url, ":\"", y_absl::FormatTime(t), "\""));
        } else {
          kvs.push_back(y_absl::StrCat(type_url, ":\"",
                                     y_absl::CHexEscape(payload_view), "\""));
        }
      } else {
        kvs.push_back(y_absl::StrCat(type_url, ":\"",
                                   y_absl::CHexEscape(payload_view), "\""));
      }
    } else {
      y_absl::optional<y_absl::string_view> payload_view = payload.TryFlat();
      TString payload_str = y_absl::CHexEscape(
          payload_view.has_value() ? *payload_view : TString(payload));
      kvs.push_back(y_absl::StrCat(type_url, ":\"", payload_str, "\""));
    }
  });
  if (children.has_value()) {
    std::vector<y_absl::Status> children_status = ParseChildren(*children);
    std::vector<TString> children_text;
    children_text.reserve(children_status.size());
    for (const y_absl::Status& child_status : children_status) {
      children_text.push_back(StatusToString(child_status));
    }
    kvs.push_back(
        y_absl::StrCat("children:[", y_absl::StrJoin(children_text, ", "), "]"));
  }
  return kvs.empty() ? head
                     : y_absl::StrCat(head, " {", y_absl::StrJoin(kvs, ", "), "}");
}

namespace internal {

google_rpc_Status* StatusToProto(const y_absl::Status& status, upb_Arena* arena) {
  google_rpc_Status* msg = google_rpc_Status_new(arena);
  google_rpc_Status_set_code(msg, static_cast<int32_t>(status.code()));
  // Protobuf string field requires to be utf-8 encoding but C++ string doesn't
  // this requirement so it can be a non utf-8 string. So it should be converted
  // to a percent-encoded string to keep it as a utf-8 string.
  Slice message_percent_slice =
      PercentEncodeSlice(Slice::FromExternalString(status.message()),
                         PercentEncodingType::Compatible);
  char* message_percent = reinterpret_cast<char*>(
      upb_Arena_Malloc(arena, message_percent_slice.length()));
  if (message_percent_slice.length() > 0) {
    memcpy(message_percent, message_percent_slice.data(),
           message_percent_slice.length());
  }
  google_rpc_Status_set_message(
      msg, upb_StringView_FromDataAndSize(message_percent,
                                          message_percent_slice.length()));
  status.ForEachPayload([&](y_absl::string_view type_url,
                            const y_absl::Cord& payload) {
    google_protobuf_Any* any = google_rpc_Status_add_details(msg, arena);
    char* type_url_buf =
        reinterpret_cast<char*>(upb_Arena_Malloc(arena, type_url.size()));
    memcpy(type_url_buf, type_url.data(), type_url.size());
    google_protobuf_Any_set_type_url(
        any, upb_StringView_FromDataAndSize(type_url_buf, type_url.size()));
    y_absl::optional<y_absl::string_view> v_view = payload.TryFlat();
    if (v_view.has_value()) {
      google_protobuf_Any_set_value(
          any, upb_StringView_FromDataAndSize(v_view->data(), v_view->size()));
    } else {
      char* buf =
          reinterpret_cast<char*>(upb_Arena_Malloc(arena, payload.size()));
      char* cur = buf;
      for (y_absl::string_view chunk : payload.Chunks()) {
        memcpy(cur, chunk.data(), chunk.size());
        cur += chunk.size();
      }
      google_protobuf_Any_set_value(
          any, upb_StringView_FromDataAndSize(buf, payload.size()));
    }
  });
  return msg;
}

y_absl::Status StatusFromProto(google_rpc_Status* msg) {
  int32_t code = google_rpc_Status_code(msg);
  upb_StringView message_percent_upb = google_rpc_Status_message(msg);
  Slice message_percent_slice = Slice::FromExternalString(
      y_absl::string_view(message_percent_upb.data, message_percent_upb.size));
  Slice message_slice =
      PermissivePercentDecodeSlice(std::move(message_percent_slice));
  y_absl::Status status(
      static_cast<y_absl::StatusCode>(code),
      y_absl::string_view(reinterpret_cast<const char*>(message_slice.data()),
                        message_slice.size()));
  size_t detail_len;
  const google_protobuf_Any* const* details =
      google_rpc_Status_details(msg, &detail_len);
  for (size_t i = 0; i < detail_len; i++) {
    upb_StringView type_url = google_protobuf_Any_type_url(details[i]);
    upb_StringView value = google_protobuf_Any_value(details[i]);
    status.SetPayload(y_absl::string_view(type_url.data, type_url.size),
                      y_absl::Cord(y_absl::string_view(value.data, value.size)));
  }
  return status;
}

uintptr_t StatusAllocHeapPtr(y_absl::Status s) {
  if (s.ok()) return 0;
  y_absl::Status* ptr = new y_absl::Status(s);
  return reinterpret_cast<uintptr_t>(ptr);
}

void StatusFreeHeapPtr(uintptr_t ptr) {
  y_absl::Status* s = reinterpret_cast<y_absl::Status*>(ptr);
  delete s;
}

y_absl::Status StatusGetFromHeapPtr(uintptr_t ptr) {
  if (ptr == 0) {
    return y_absl::OkStatus();
  } else {
    return *reinterpret_cast<y_absl::Status*>(ptr);
  }
}

y_absl::Status StatusMoveFromHeapPtr(uintptr_t ptr) {
  if (ptr == 0) {
    return y_absl::OkStatus();
  } else {
    y_absl::Status* s = reinterpret_cast<y_absl::Status*>(ptr);
    y_absl::Status ret = std::move(*s);
    delete s;
    return ret;
  }
}

}  // namespace internal

}  // namespace grpc_core
