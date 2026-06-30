#pragma once

#include <google/protobuf/stubs/port.h>
#include <google/protobuf/io/coded_stream.h>
#include <util/stream/output.h>
#include <util/generic/buffer.h>

#include <google/protobuf/io/zero_copy_stream_impl.h>

/// this file is Yandex extensions to protobuf

namespace google {
namespace protobuf {

class Message;

namespace internal {

struct TAsBinary {
  const Message& Message_;
  friend IOutputStream& operator <<(IOutputStream& output, const TAsBinary& wrappedMessage);
};

struct TAsStreamSeq {
  const Message& Message_;
  friend IOutputStream& operator <<(IOutputStream& output, const TAsStreamSeq& wrappedMessage);
};

} // namespace internal

namespace io {

/// Parse*Seq methods read message size from stream to find a message boundary

/// there is not parse from IInputStream, because it is not push-backable

bool ParseFromCodedStreamSeq(Message* msg, io::CodedInputStream* input);
bool ParseFromZeroCopyStreamSeq(Message* msg, io::ZeroCopyInputStream* input);

/// Serialize*Seq methods write message size as varint before writing a message
/// there is no serialize to IOutputStream, because it is not push-backable

bool SerializePartialToCodedStreamSeq(const Message* msg, io::CodedOutputStream* output);
bool SerializeToCodedStreamSeq(const Message* msg, io::CodedOutputStream* output);
bool SerializeToZeroCopyStreamSeq(const Message* msg, io::ZeroCopyOutputStream* output);

class TErrorState {
public:
    TErrorState()
        : HasError_(false)
    {
    }
    bool HasError() const {
        return HasError_;
    }
    void SetError() {
        HasError_ = true;
    }
private:
    bool HasError_;
};

class TInputStreamProxy: public io::CopyingInputStream, public TErrorState {
    public:
        inline TInputStreamProxy(IInputStream* slave)
            : mSlave(slave)
        {
        }

        virtual int Read(void* buffer, int size);

    private:
        IInputStream* mSlave;
};

class TOutputStreamProxy: public io::CopyingOutputStream, public TErrorState {
    public:
        inline TOutputStreamProxy(IOutputStream* slave)
            : mSlave(slave)
        {
        }

        virtual bool Write(const void* buffer, int size);

    private:
        IOutputStream* mSlave;
};


class TCopyingInputStreamAdaptor: public TInputStreamProxy, public CopyingInputStreamAdaptor {
public:
    TCopyingInputStreamAdaptor(IInputStream* inputStream)
        : TInputStreamProxy(inputStream)
        , CopyingInputStreamAdaptor(this)
    { }
};

class TCopyingOutputStreamAdaptor: public TOutputStreamProxy, public CopyingOutputStreamAdaptor {
public:
    TCopyingOutputStreamAdaptor(IOutputStream* outputStream)
        : TOutputStreamProxy(outputStream)
        , CopyingOutputStreamAdaptor(this)
    { }
};


class TProtoSerializer {
public:
    static void Save(IOutputStream* output, const Message& msg);
    static void Load(IInputStream* input, Message& msg);
    static void MergeFrom(IInputStream* input, Message& msg);

    // similar interface for protobuf coded streams
    static inline bool Save(CodedOutputStream* output, const Message& msg) {
        return SerializeToCodedStreamSeq(&msg, output);
    }

    static inline bool Load(CodedInputStream* input, Message& msg) {
        return ParseFromCodedStreamSeq(&msg, input);
    }
};


/**
 * Special separate simple reader of protobuf files from arcadic streams
 * with static one time allocated buffer.
 *
 * Data can be prepared with TProtoSerializer::Save, format is the same.
 *
 */
class TProtoReader {
public:
    TProtoReader(IInputStream* input, const size_t bufferSize = DefaultBufferSize);

    /**
     * Reads protobuf message
     *
     * @param msg       binary compatible protobuf message
     * @returns         true  if read is ok
     *                  false if stream is exhausted
     *                  yexception if input data is corrupted
     */
    bool Load(Message& msg);

private:
    IInputStream* IStream;
    TBuffer       Buffer;

    static const size_t DefaultBufferSize = (1 << 16);
};

} // namespace io
} // namespace protobuf
} // namespace google

// arcadia-style serialization
inline void Save(IOutputStream* output, const google::protobuf::Message& msg) {
    google::protobuf::io::TProtoSerializer::Save(output, msg);
}

inline void Load(IInputStream* input, google::protobuf::Message& msg) {
    google::protobuf::io::TProtoSerializer::Load(input, msg);
}

// A mix of ShortDebugString and Utf8DebugString
TProtoStringType ShortUtf8DebugString(const google::protobuf::Message& msg);
