#pragma once

#include <util/generic/strbuf.h>
#include <util/stream/output.h>
#include <util/string/ascii.h>

namespace NNeh {
    namespace NHttp {
        template <typename Port>
        void WriteHostHeader(IOutputStream& out, TStringBuf host, Port port) {
            out << TStringBuf("Host: ") << host;
            if (port) {
                out << TStringBuf(":") << port;
            }
            out << TStringBuf("\r\n");
        }

        class THeaderSplitter {
        public:
            THeaderSplitter(TStringBuf headers)
                : Headers_(headers)
            {
            }

            bool Next(TStringBuf& header) {
                while (Headers_.ReadLine(header)) {
                    if (!header.empty()) {
                        return true;
                    }
                }
                return false;
            }
        private:
            TStringBuf Headers_;
        };

        inline bool HasHostHeader(TStringBuf headers) {
            THeaderSplitter splitter(headers);
            TStringBuf header;
            while (splitter.Next(header)) {
                if (AsciiHasPrefixIgnoreCase(header, "Host:")) {
                    return true;
                }
            }
            return false;
        }

        template <typename Port>
        void WriteHostHeaderIfNot(IOutputStream& out, TStringBuf host, Port port, TStringBuf headers) {
            if (!NNeh::NHttp::HasHostHeader(headers)) {
                NNeh::NHttp::WriteHostHeader(out, host, port);
            }
        }
    }
}
