#pragma once


class IOutputStream;

namespace google {
    namespace protobuf {
        class Message;
    }
}

namespace NProtoBufInternal {
  struct TAsBinary {
    const google::protobuf::Message& Message_;
    friend IOutputStream& operator <<(IOutputStream& output, const TAsBinary& wrappedMessage);
  };

  struct TAsStreamSeq {
    const google::protobuf::Message& Message_;
    friend IOutputStream& operator <<(IOutputStream& output, const TAsStreamSeq& wrappedMessage);
  };
}
