#pragma once

#include <util/generic/string.h>
#include <util/generic/strbuf.h>
#include <util/generic/yexception.h>
#include <util/generic/hash_set.h>
#include <util/string/cast.h>
#include <library/cpp/http/io/stream.h>

struct THttpVersion {
    unsigned Major = 1;
    unsigned Minor = 0;
};

//http requests parser for async/callbacks arch. (uggly state-machine)
//usage, - call Parse(...), if returned 'true' - all message parsed,
//external (non entered in message) bytes in input data counted by GetExtraDataSize()
class THttpParser {
public:
    enum TMessageType {
        Request,
        Response
    };

    THttpParser(TMessageType mt = Response)
        : Parser_(&THttpParser::FirstLineParser)
        , MessageType_(mt)
    {
    }

    inline void DisableCollectingHeaders() noexcept {
        CollectHeaders_ = false;
    }

    inline void SetGzipAllowMultipleStreams(bool allow) noexcept {
        GzipAllowMultipleStreams_ = allow;
    }

    inline void DisableDecodeContent() noexcept {
        DecodeContent_ = false;
    }

    /*
     * Disable message-body parsing.
     * Useful for parse HEAD method responses
     */
    inline void BodyNotExpected() {
        BodyNotExpected_ = true;
    }

    /// @return true on end parsing (GetExtraDataSize() return amount not used bytes)
    /// throw exception on bad http format (unsupported encoding, etc)
    /// sz == 0 signaling end of input stream
    bool Parse(const char* data, size_t sz) {
        if (ParseImpl(data, sz)) {
            if (DecodeContent_) {
                DecodeContent(DecodedContent_);
            }
            return true;
        }
        return false;
    }

    const char* Data() const noexcept {
        return Data_;
    }

    size_t GetExtraDataSize() const noexcept {
        return ExtraDataSize_;
    }

    const TString& FirstLine() const noexcept {
        return FirstLine_;
    }

    unsigned RetCode() const noexcept {
        return RetCode_;
    }

    const THttpVersion& HttpVersion() const noexcept {
        return HttpVersion_;
    }

    const THttpHeaders& Headers() const noexcept {
        return Headers_;
    }

    bool IsKeepAlive() const noexcept {
        return KeepAlive_;
    }

    bool GetContentLength(ui64& value) const noexcept {
        if (!HasContentLength_) {
            return false;
        }

        value = ContentLength_;
        return true;
    }

    TString GetBestCompressionScheme() const;

    const THashSet<TString>& AcceptedEncodings() const;

    const TString& Content() const noexcept {
        return Content_;
    }

    const TString& DecodedContent() const noexcept {
        return DecodedContent_;
    }

    void Prepare() {
        HeaderLine_.reserve(128);
        FirstLine_.reserve(128);
    }

    bool DecodeContent(TString& decodedContent) const;

private:
    bool ParseImpl(const char* data, size_t sz) {
        Data_ = data;
        DataEnd_ = data + sz;
        if (sz == 0) {
            OnEof();
            return true;
        }
        return (this->*Parser_)();
    }
    // stage parsers
    bool FirstLineParser();
    bool HeadersParser();
    bool ContentParser();
    bool ChunkedContentParser();
    bool OnEndParsing();

    // continue read to CurrentLine_
    bool ReadLine();

    void ParseHttpVersion(TStringBuf httpVersion);
    void ParseHeaderLine();

    void OnEof();

    void ApplyHeaderLine(const TStringBuf& name, const TStringBuf& val);

    typedef bool (THttpParser::*TParser)();

    TParser Parser_; //current parser (stage)
    TMessageType MessageType_ = Response;
    bool CollectHeaders_ = true;
    bool GzipAllowMultipleStreams_ = true;
    bool DecodeContent_ = true;
    bool BodyNotExpected_ = false;

    // parsed data
    const char* Data_ = nullptr;
    const char* DataEnd_ = nullptr;
    TString CurrentLine_;
    TString HeaderLine_;

    size_t ExtraDataSize_ = 0;

    // headers
    TString FirstLine_;
    THttpVersion HttpVersion_;
    unsigned RetCode_ = 0;
    THttpHeaders Headers_;
    bool KeepAlive_ = false;
    THashSet<TString> AcceptEncodings_;

    TString ContentEncoding_;
    bool HasContentLength_ = false;
    ui64 ContentLength_ = 0;

    struct TChunkInputState {
        size_t LeftBytes_ = 0;
        bool ReadLastChunk_ = false;
    };

    TAutoPtr<TChunkInputState> ChunkInputState_;

    TString Content_;
    TString DecodedContent_;
};
