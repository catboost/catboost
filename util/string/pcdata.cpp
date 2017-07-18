#include "strspn.h"
#include "pcdata.h"

static TCompactStrSpn sspn("\"<>&'");

static void EncodeHtmlPcdataAppendInternal(const TStringBuf str, TString& strout, bool qAmp) {
    const char* s = str.data();
    const char* e = s + str.length();

    for (;;) {
        const char* next = sspn.FindFirstOf(s, e);

        strout.AppendNoAlias(s, next - s);
        s = next;

        if (s == e)
            break;

        switch (*s) {
            case '\"':
                strout += STRINGBUF("&quot;");
                ++s;
                break;

            case '<':
                strout += STRINGBUF("&lt;");
                ++s;
                break;

            case '>':
                strout += STRINGBUF("&gt;");
                ++s;
                break;

            case '\'':
                strout += STRINGBUF("&#39;");
                ++s;
                break;

            case '&':
                if (qAmp)
                    strout += STRINGBUF("&amp;");
                else
                    strout += STRINGBUF("&");
                ++s;
                break;
        }
    }
}

void EncodeHtmlPcdataAppend(const TStringBuf str, TString& strout) {
    EncodeHtmlPcdataAppendInternal(str, strout, true);
}

TString EncodeHtmlPcdata(const TStringBuf str, bool qAmp) {
    TString strout;
    EncodeHtmlPcdataAppendInternal(str, strout, qAmp);
    return strout;
}

TString DecodeHtmlPcdata(const TString& sz) {
    TString res;
    const char* codes[] = {"&quot;", "&lt;", "&gt;", "&#39;", "&#039;", "&amp;", "&apos;", nullptr};
    const char chars[] = {'\"', '<', '>', '\'', '\'', '&', '\''};
    for (size_t i = 0; i < sz.length(); ++i) {
        char c = sz[i];
        if (c == '&') {
            for (const char** p = codes; *p; ++p) {
                size_t len = strlen(*p);
                if (strncmp(sz.c_str() + i, *p, len) == 0) {
                    i += len - 1;
                    c = chars[p - codes];
                    break;
                }
            }
        }
        res += c;
    }
    return res;
}
