#include "pcre2pire.h"
#include <util/generic/vector.h>
#include <util/generic/yexception.h>

TString Pcre2Pire(const TString& src) {
    TVector<char> result;
    result.reserve(src.size() + 1);

    enum EState {
        S_SIMPLE,
        S_SLASH,
        S_BRACE,
        S_EXPECT_Q,
        S_QUESTION,
        S_P,
        S_COMMA,
        S_IN,
    };

    EState state = S_SIMPLE;

    for (ui32 i = 0; i < src.size(); ++i) {
        const char c = src[i];

        switch (state) {
            case S_SIMPLE:
                if (c == '\\') {
                    state = S_SLASH;
                } else if (c == '(') {
                    state = S_BRACE;
                } else if (c == '*' || c == '?') {
                    state = S_EXPECT_Q;
                    result.push_back(c);
                } else {
                    if (c == ')' && result.size() > 0 && result.back() == '(') {
                        // eliminating "()"
                        result.pop_back();
                    } else {
                        result.push_back(c);
                    }
                }
                break;
            case S_SLASH:
                state = S_SIMPLE;
                if (c == ':' || c == '=' || c == '#' || c == '&') {
                    result.push_back(c);
                } else {
                    result.push_back('\\');
                    --i;
                }
                break;
            case S_BRACE:
                if (c == '?') {
                    state = S_QUESTION;
                } else {
                    state = S_COMMA;
                    --i;
                }
                break;
            case S_EXPECT_Q:
                state = S_SIMPLE;
                if (c != '?') {
                    --i;
                }
                break;
            case S_QUESTION:
                if (c == 'P') {
                    state = S_P;
                } else if (c == ':' || c == '=') {
                    state = S_COMMA;
                } else {
                    ythrow yexception() << "Pcre to pire convertaion failed: unexpected symbol '" << c << "' at posiotion " << i << "!";
                }
                break;
            case S_P:
                if (c == '<') {
                    state = S_IN;
                } else {
                    ythrow yexception() << "Pcre to pire convertaion failed: unexpected symbol '" << c << "' at posiotion " << i << "!";
                }
                break;
            case S_IN:
                if (c == '>') {
                    state = S_COMMA;
                } else {
                    // nothing to do
                }
                break;
            case S_COMMA:
                state = S_SIMPLE;
                if (c == ')') {
                    // nothing to do
                } else {
                    result.push_back('(');
                    --i;
                }
                break;
            default:
                ythrow yexception() << "Pcre to pire convertaion failed: unexpected automata state!";
        }
    }

    if (state != S_SIMPLE && state != S_EXPECT_Q) {
        ythrow yexception() << "Pcre to pire convertaion failed: unexpected end of expression!";
    }

    result.push_back('\0');

    return &result[0];
}
