#include "headers.h"
#include "stream.h"

#include <util/generic/yexception.h>
#include <util/generic/strbuf.h>
#include <util/string/cast.h>
#include <util/stream/output.h>
#include <util/string/strip.h>

static inline TStringBuf Trim(const char* b, const char* e) noexcept {
    return StripString(TStringBuf(b, e));
}

THttpInputHeader::THttpInputHeader(const TString& header) {
    size_t pos = header.find(':');

    if (pos == TString::npos) {
        ythrow THttpParseException() << "can not parse http header(" << header.Quote() << ")";
    }

    Name_ = TString(header.begin(), header.begin() + pos);
    Value_ = ::ToString(Trim(header.begin() + pos + 1, header.end()));
}

THttpInputHeader::THttpInputHeader(const TString& name, const TString& value)
    : Name_(name)
    , Value_(value)
{
}

THttpInputHeader::~THttpInputHeader() {
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

THttpHeaders::THttpHeaders() {
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
            AddHeader(header);
            header = line;
        }
    }
}

bool THttpHeaders::HasHeader(const TString& header) const {
    for (THeaders::const_iterator h = Headers_.begin(); h != Headers_.end(); ++h) {
        if (stricmp(~h->Name(), ~header) == 0) {
            return true;
        }
    }
    return false;
}

void THttpHeaders::RemoveHeader(const TString& header) {
    for (THeaders::iterator h = Headers_.begin(); h != Headers_.end(); ++h) {
        if (stricmp(~h->Name(), ~header) == 0) {
            Headers_.erase(h);
            return;
        }
    }
}

void THttpHeaders::AddOrReplaceHeader(const THttpInputHeader& header) {
    for (auto& Header : Headers_) {
        if (stricmp(~Header.Name(), ~header.Name()) == 0) {
            Header = header;

            return;
        }
    }

    AddHeader(header);
}

void THttpHeaders::AddHeader(const THttpInputHeader& header) {
    Headers_.push_back(header);
}

THttpHeaders::~THttpHeaders() {
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
