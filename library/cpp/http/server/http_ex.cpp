#include "http_ex.h"

#include <util/generic/buffer.h>
#include <util/generic/cast.h>
#include <util/stream/null.h>

bool THttpClientRequestExtension::Parse(char* req, TBaseServerRequestData& rd) {
    rd.SetSocket(Socket());

    if (!rd.Parse(req)) {
        Output() << "HTTP/1.1 403 Forbidden\r\n"
                    "Content-Type: text/plain\r\n"
                    "Content-Length: 39\r\n"
                    "\r\n"
                    "The server cannot be used as a proxy.\r\n";

        return false;
    }

    return true;
}

bool THttpClientRequestExtension::ProcessHeaders(TBaseServerRequestData& rd, TBlob& postData) {
    for (const auto& header : ParsedHeaders) {
        rd.AddHeader(header.first, header.second);
    }

    char* s = RequestString.begin();

    enum EMethod {
        NotImplemented,
        Get,
        Post,
        Put,
        Patch,
        Delete,
        Options,
    };

    enum EMethod foundMethod;
    char* urlStart;

    if (strnicmp(s, "GET ", 4) == 0) {
        foundMethod = Get;
        urlStart = s + 4;
    } else if (strnicmp(s, "POST ", 5) == 0) {
        foundMethod = Post;
        urlStart = s + 5;
    } else if (strnicmp(s, "PUT ", 4) == 0) {
        foundMethod = Put;
        urlStart = s + 4;
    } else if (strnicmp(s, "PATCH ", 6) == 0) {
        foundMethod = Patch;
        urlStart = s + 6;
    } else if (strnicmp(s, "DELETE ", 7) == 0) {
        foundMethod = Delete;
        urlStart = s + 7;
    } else if (strnicmp(s, "OPTIONS ", 8) == 0) {
        foundMethod = Options;
        urlStart = s + 8;
    } else {
        foundMethod = NotImplemented;
    }

    switch (foundMethod) {
        case Get:
        case Delete:
            if (!Parse(urlStart, rd)) {
                return false;
            }
            break;

        case Post:
        case Put:
        case Patch:
            try {
                ui64 contentLength = 0;
                if (Input().HasExpect100Continue()) {
                    Output().SendContinue();
                }

                if (!Input().ContentEncoded() && Input().GetContentLength(contentLength)) {
                    if (contentLength > HttpServ()->Options().MaxInputContentLength) {
                        Output() << "HTTP/1.1 413 Payload Too Large\r\nContent-Length:0\r\n\r\n";
                        Output().Finish();
                        return false;
                    }

                    TBuffer buf(SafeIntegerCast<size_t>(contentLength));
                    buf.Resize(Input().Load(buf.Data(), (size_t)contentLength));
                    postData = TBlob::FromBuffer(buf);
                } else {
                    postData = TBlob::FromStream(Input());
                }
            } catch (...) {
                Output() << "HTTP/1.1 400 Bad request\r\n\r\n";
                return false;
            }

            if (!Parse(urlStart, rd)) {
                return false;
            }
            break;

        case Options:
            if (!OptionsAllowed()) {
                Output() << "HTTP/1.1 405 Method Not Allowed\r\n\r\n";
                return false;
            } else if (!Parse(urlStart, rd)) {
                return false;
            }
            break;

        case NotImplemented:
            Output() << "HTTP/1.1 501 Not Implemented\r\n\r\n";
            return false;
    }

    return true;
}
