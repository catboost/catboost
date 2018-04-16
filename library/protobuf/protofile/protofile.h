#pragma once

// Reads and writes files with records comprised of ProtocolMessages.
#include <contrib/libs/protobuf/text_format.h>
#include <contrib/libs/protobuf/io/zero_copy_stream_impl.h>
#include <util/generic/buffer.h>
#include <util/generic/yexception.h>
#include <util/stream/file.h>

#include <util/folder/path.h>
#include <library/logger/global/global.h>
#include <errno.h>

namespace NFastTier {
    // Common interface for Proto readers
    template <class T>
    class IProtoReader {
    public:
        virtual ~IProtoReader() {
        }
        virtual void Open(IInputStream*) = 0; // Stream is not owned
        virtual void Open(const TString&) = 0;
        virtual bool GetNext(T& record) = 0;
    };

    // Common interface for Proto writers
    template <class T>
    class IProtoWriter {
    public:
        virtual ~IProtoWriter() {
        }
        virtual void Open(IOutputStream*) = 0; // Stream is not owned
        virtual void Open(const TString&) = 0;
        virtual void Write(const T&) = 0;
        virtual void Finish() = 0;
    };

    // Proto reader in bin format
    template <class T>
    class TBinaryProtoReader: public IProtoReader<T> {
    private:
        THolder<IInputStream> StreamHolder;
        IInputStream* Stream;
        size_t Position;
        TBuffer Buffer;

        bool ReadDump(void* ptr, size_t size) {
            size_t bytesLoaded = Stream->Load(ptr, size);
            Position += bytesLoaded;
            return bytesLoaded == size;
        }

    public:
        TBinaryProtoReader()
            : Stream(nullptr)
        {
        }

        void Open(IInputStream* stream) override {
            Stream = stream;
            Position = 0;
        }

        void Open(const TString& fileName) override {
            StreamHolder.Reset(new TIFStream(fileName));
            Stream = StreamHolder.Get();
            Position = 0;
        }

        size_t Tell() const {
            return Position;
        }

        void ForwardSeek(size_t newPosition) {
            Y_ENSURE(Stream, "Stream must be open. ");
            if (Position != newPosition) {
                Y_ENSURE(newPosition >= Position,
                         "Current position in the stream (" << Position << ") is past the requested one (" << newPosition << ") in ForwardSeek. ");
                Stream->Skip(newPosition - Position);
                Position = newPosition;
            }
        }

        bool GetNext(T& record) override {
            Y_ENSURE(Stream, "Stream must be open. ");
            ui32 recordSize = 0;
            if (!ReadDump(&recordSize, sizeof(ui32))) {
                return false;
            }
            Buffer.Resize(recordSize);
            Y_ENSURE(ReadDump(~Buffer, recordSize), "Corrupted record in protofile. ");
            record.Clear();
            Y_ENSURE(record.ParseFromArray(~Buffer, recordSize),
                     "Corrupted record in protofile. ");
            return true;
        }
    };

    template <class T>
    class TBinaryProtoWriter: public IProtoWriter<T> {
    private:
        THolder<IOutputStream> StreamHolder;
        IOutputStream* Stream;
        size_t Position;

        void WriteDump(void* ptr, size_t size) {
            Stream->Write(ptr, size);
            Position += size;
        }

    public:
        TBinaryProtoWriter()
            : Stream(nullptr)
            , Position(0)
        {
        }

        void Open(IOutputStream* stream) override {
            Stream = stream;
            Position = 0;
        }

        void Open(const TString& fileName) override {
            StreamHolder.Reset(new TOFStream(fileName));
            Stream = StreamHolder.Get();
            Position = 0;
        }

        size_t Tell() const {
            return Position;
        }

        void Write(const T& record) override {
            Y_ENSURE(Stream, "Stream must be open. ");
            ui32 recordSize = record.ByteSize();
            WriteDump(&recordSize, sizeof(ui32));
            Y_ENSURE(record.SerializeToStream(Stream), "Failed to serialize record");
            Position += recordSize;
        }

        void Finish() override {
            Y_ENSURE(Stream, "Stream must be open. ");
            Stream->Finish();
            StreamHolder.Destroy();
        }
    };

    class TOutputStreamAdaptor: public google::protobuf::io::CopyingOutputStream {
    private:
        IOutputStream* Stream;

    public:
        TOutputStreamAdaptor(IOutputStream* outstream)
            : Stream(outstream)
        {
        }

        bool Write(const void* buffer, int size) override {
            Stream->Write(buffer, size);
            Stream->Flush();
            return true;
        }
    };

    template <class T>
    class TTextProtoReader: public IProtoReader<T> {
    private:
        THolder<IInputStream> TextInputHolder;
        IInputStream* TextInput;

    public:
        TTextProtoReader()
            : TextInput(nullptr)
        {
        }

        void Open(IInputStream* stream) override {
            TextInput = stream;
        }

        void Open(const TString& file) override {
            TextInputHolder.Reset(new TIFStream(file));
            TextInput = TextInputHolder.Get();
        }

        bool GetNext(T& record) override {
            Y_ENSURE(TextInput, "Stream must be open. ");
            TString tmpString;
            TString line;
            bool lastReadSuccess;
            while ((lastReadSuccess = TextInput->ReadLine(line))) {
                if (line.Empty()) {
                    break;
                }
                tmpString += line;
                tmpString.push_back('\n');
            }
            if (!lastReadSuccess && tmpString.Empty()) {
                return false;
            }
            google::protobuf::TextFormat::ParseFromString(tmpString, &record);
            return true;
        }
    };

    template <class T>
    class TTextProtoWriter: public IProtoWriter<T> {
    private:
        THolder<IOutputStream> TextOutputHolder;
        IOutputStream* TextOutput;

    public:
        TTextProtoWriter()
            : TextOutput(nullptr)
        {
        }

        void Open(IOutputStream* stream) override {
            TextOutput = stream;
        }

        void Open(const TString& file) override {
            TextOutputHolder.Reset(new TOFStream(file));
            TextOutput = TextOutputHolder.Get();
        }

        void Write(const T& record) override {
            Y_ENSURE(TextOutput, "Stream must be open. ");
            TString tmpString;
            google::protobuf::TextFormat::PrintToString(record, &tmpString);
            *TextOutput << tmpString << AsStringBuf("\n"); // Extra Endline is the record separator, don't use Endl to avoid flushing.
        }

        void Finish() override {
            Y_ENSURE(TextOutput, "Stream must be open. ");
            TextOutput->Flush();
            TextOutputHolder.Destroy();
        }
    };

}

template <class TProto>
class TProtoFileGuard {
public:
    TProtoFileGuard(const TFsPath& path, bool forceOpen = false)
        : Path(path)
        , Changed(false)
    {
        if (!path.Exists()) {
            return;
        }

        TFileInput fi(path);
        if (forceOpen) {
            if (!::google::protobuf::TextFormat::ParseFromString(fi.ReadAll(), &Proto)) {
                path.ForceDelete();
            }
        } else {
            VERIFY_WITH_LOG(::google::protobuf::TextFormat::ParseFromString(fi.ReadAll(), &Proto), "Corrupted %s", ~Path.GetPath());
        }
    }

    TProto* operator->() {
        Changed = true;
        return &Proto;
    }

    const TProto* operator->() const {
        return &Proto;
    }

    TProto& operator*() {
        Changed = true;
        return Proto;
    }

    const TProto& operator*() const {
        return Proto;
    }

    const TProto& GetProto() const {
        return Proto;
    }

    void Flush() {
        if (!Changed)
            return;
        try {
            TString out;
            VERIFY_WITH_LOG(::google::protobuf::TextFormat::PrintToString(Proto, &out), "Error while serializing %s", ~Path.GetPath());
            TFsPath tmpPath = Path.Parent() / ("~" + Path.GetName());
            {
                TUnbufferedFileOutput fo(tmpPath);
                fo.Write(~out, +out);
            }
            tmpPath.ForceRenameTo(Path);
        } catch (...) {
            FAIL_LOG("cannot save %s: %s, errno = %i", ~Path.GetPath(), ~CurrentExceptionMessage(), errno);
        }
    }

    ~TProtoFileGuard() {
        Flush();
    }

private:
    TProto Proto;
    TFsPath Path;
    bool Changed;
};
