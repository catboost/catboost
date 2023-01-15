#ifndef GOOGLE_PROTOBUF_MESSAGEINT_H__
#define GOOGLE_PROTOBUF_MESSAGEINT_H__

#include <contrib/libs/protobuf/io/coded_stream.h>
#include <util/stream/output.h>
#include <descriptor.h>
#include <stubs/substitute.h>

namespace google {
namespace protobuf {

static string InitializationErrorMessage(const char* action,
                                         const Message& message) {
  return strings::Substitute(
    "Can't $0 message of type \"$1\" because it is missing required "
    "fields: $2",
    action, message.GetDescriptor()->full_name(),
    message.InitializationErrorString());
  return "";
}

}
}

#endif
