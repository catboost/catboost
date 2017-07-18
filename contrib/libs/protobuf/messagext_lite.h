#pragma once


class TOutputStream;

namespace google {
    namespace protobuf {
        class Message;
    }
}

namespace NProtoBufInternal {
  struct TAsBinary {
    const google::protobuf::Message& Message_;
    friend TOutputStream& operator <<(TOutputStream& output, const TAsBinary& wrappedMessage);
  };

  struct TAsStreamSeq {
    const google::protobuf::Message& Message_;
    friend TOutputStream& operator <<(TOutputStream& output, const TAsStreamSeq& wrappedMessage);
  };
}
