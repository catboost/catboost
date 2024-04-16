#include "headers.h"
#include "stream.h"

#include <util/generic/strbuf.h>
#include <util/generic/yexception.h>
#include <util/stream/output.h>
#include <util/string/ascii.h>
#include <util/string/cast.h>
#include <util/string/strip.h>

static inline TStringBuf Trim(const char* b, const char* e) noexcept {
    return StripString(TStringBuf(b, e));
}

static inline bool HeaderNameEqual(TStringBuf headerName, TStringBuf expectedName) noexcept {
    // Most headers names have distinct sizes.
    // Size comparison adds small overhead if all headers have the same size (~4% or lower with size = 4),
    // but significantly speeds up the case where sizes are different (~4.5x for expectedName.size() = 4 and headerName.size() = 5)
    return headerName.size() == expectedName.size() && AsciiCompareIgnoreCase(headerName, expectedName) == 0;
}

THttpInputHeader::THttpInputHeader(const TStringBuf header) {
    size_t pos = header.find(':');

    if (pos == TString::npos) {
        ythrow THttpParseException() << "can not parse http header(" << TString{header}.Quote() << ")";
    }

    Name_ = TString(header.cbegin(), header.cbegin() + pos);
    Value_ = ::ToString(Trim(header.cbegin() + pos + 1, header.cend()));
}

THttpInputHeader::THttpInputHeader(TString name, TString value)
    : Name_(std::move(name))
    , Value_(std::move(value))
{
}

void THttpInputHeader::OutTo(IOutputStream* stream) const {
    typedef IOutputStream::TPart TPart;

    const TPart parts[] = {
        TPart(Name_),
        TPart(": ", 2),
        TPart(Value_),
        TPart::CrLf(),
    };

    stream->Write(parts, sizeof(parts) / sizeof(*parts));
}

THttpHeaders::THttpHeaders(IInputStream* stream) {
    TString header;
    TString line;

    bool rdOk = stream->ReadLine(header);
    while (rdOk && !header.empty()) {
        rdOk = stream->ReadLine(line);

        if (rdOk && ((line[0] == ' ') || (line[0] == '\t'))) {
            header += line;
        } else {
            AddHeader(THttpInputHeader(header));
            header = line;
        }
    }
}

THttpHeaders::THttpHeaders(TArrayRef<const THttpInputHeader> headers) {
    for (const auto& header : headers) {
        AddHeader(header);
    }
}


bool THttpHeaders::HasHeader(const TStringBuf header) const {
    return FindHeader(header);
}

const THttpInputHeader* THttpHeaders::FindHeader(const TStringBuf header) const {
    for (const auto& hdr : Headers_) {
        if (HeaderNameEqual(hdr.Name(), header)) {
            return &hdr;
        }
    }
    return nullptr;
}

void THttpHeaders::RemoveHeader(const TStringBuf header) {
    for (auto h = Headers_.begin(); h != Headers_.end(); ++h) {
        if (HeaderNameEqual(h->Name(), header)) {
            Headers_.erase(h);
            return;
        }
    }
}

void THttpHeaders::AddOrReplaceHeader(const THttpInputHeader& header) {
    TStringBuf name = header.Name();
    for (auto& hdr : Headers_) {
        if (HeaderNameEqual(hdr.Name(), name)) {
            hdr = header;
            return;
        }
    }

    AddHeader(header);
}

void THttpHeaders::AddHeader(THttpInputHeader header) {
    Headers_.push_back(std::move(header));
}

void THttpHeaders::OutTo(IOutputStream* stream) const {
    for (TConstIterator header = Begin(); header != End(); ++header) {
        header->OutTo(stream);
    }
}

template <>
void Out<THttpHeaders>(IOutputStream& out, const THttpHeaders& h) {
    h.OutTo(&out);
}
