#include "nlptypes.h"
#include "token_structure.h"

template <typename TChr>
static NLP_TYPE GuessTypeByWordT(const TChr* w, size_t len) {
    // NLP_WORD
    // NLP_INTEGER
    // NLP_FLOAT
    // NLP_MARK

    //integer         {digit}+
    //fixed           {digit}+"."{digit}+
    enum EState {
        G_START,
        G_INT,
        G_DOT,
        G_FRAC,
    };
    EState state = G_START;

    static const TChr DIGITS[] =  {'0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 0};
    static const TBasicStringBuf<TChr> DIGITS_BUF{DIGITS};

    for (unsigned i = 0; i < len; ++i) {
        TChr c = w[i];
        bool bIsDigit = DIGITS_BUF.Contains(c);
        switch (state) {
            case G_START:
                if (bIsDigit)
                    state = G_INT;
                else {
                    if (TBasicStringBuf<TChr>(w, len).find_first_of(DIGITS_BUF) >= len)
                        return NLP_WORD;
                    else
                        return NLP_MARK;
                }
                break;
            case G_INT:
                if (bIsDigit)
                    break;
                else if (c == '.')
                    state = G_DOT;
                else
                    return NLP_MARK;
                break;
            case G_DOT:
                if (bIsDigit)
                    state = G_FRAC;
                else
                    return NLP_MARK;
                break;
            case G_FRAC:
                if (bIsDigit)
                    break;
                else
                    return NLP_MARK;
                break;
        }
    }
    switch (state) {
        case G_START:
            return NLP_MARK;
        case G_INT:
            return NLP_INTEGER;
        case G_DOT:
            return NLP_FLOAT; // NLP_MARK?
        case G_FRAC:
            return NLP_FLOAT;
    }
    Y_ASSERT(0);
    return NLP_MARK;
}

NLP_TYPE GuessTypeByWord(const char* w, unsigned len) {
    return GuessTypeByWordT(w, len);
}

NLP_TYPE GuessTypeByWord(const wchar16* w, unsigned len) {
    return GuessTypeByWordT(w, len);
}

NLP_TYPE DetectNLPType(const TTokenStructure& subtokens) {
    Y_ASSERT(subtokens.size() >= 1);
    for (size_t i = 1; i < subtokens.size(); ++i) {
        Y_ASSERT(subtokens[i].Type == TOKEN_WORD || subtokens[i].Type == TOKEN_NUMBER);
        if (subtokens[i].Type != subtokens[0].Type)
            return NLP_MARK;
    }
    if (subtokens[0].Type == TOKEN_WORD)
        return NLP_WORD;
    else if (subtokens[0].Type == TOKEN_MARK)
        return NLP_MARK;
    else if (subtokens[0].Type == TOKEN_FLOAT)
        return NLP_FLOAT;
    else if (subtokens[0].Type == TOKEN_NUMBER)
        return NLP_INTEGER;
    return NLP_MARK;
}
