#pragma once

#include <library/cpp/token/nlptypes.h>
#include <library/cpp/langmask/langmask.h>
#include <library/cpp/token/token_structure.h>

#include <util/system/defaults.h>
#include <util/generic/yexception.h>
#include <util/generic/noncopyable.h>

#include <cassert>
#include <cstdlib>

class ITokenHandler {
public:
    // Исключение, которое может кидаться обработчиком из OnToken.
    // Токенайзер проглатывает такое исключение и прекращает токенизацию
    class TAllDoneException: public yexception {
    public:
        TAllDoneException() {
            *this << "Token handler: all done";
        }
    };

    virtual void OnToken(const TWideToken& token, size_t origleng, NLP_TYPE type) = 0;
    virtual ~ITokenHandler() {
    }
};

struct TTokenizerOptions {
    bool SpacePreserve = false;
    TLangMask LangMask = TLangMask();
    bool UrlDecode = true;
    size_t Version = 2;
    bool KeepAffixes = false; // keep prefix/suffix as part of token
};

//! breaks up a text into tokens and calls to @c ITokenHandler::OnToken()
//! @note the tokenizer produces tokens of the following types only:
//!       NLP_WORD, NLP_INTEGER, NLP_FLOAT, NLP_MARK, NLP_SENTBREAK, NLP_PARABREAK, NLP_MISCTEXT.
class TNlpTokenizer: private TNonCopyable {
private:
    ITokenHandler& TokenHandler;
    const bool BackwardCompatible; //!< tokenizer reproduce old tokenization of marks
    TTempArray<wchar16> Buffer;
    const wchar16* TextStart = nullptr;

public:
    explicit TNlpTokenizer(ITokenHandler& handler, bool backwardCompatible = true)
        : TokenHandler(handler)
        , BackwardCompatible(backwardCompatible)
        , Buffer()
    {
    }

    //! the main tokenizing function
    //! @attention zero-character ('\0') considered as word break, so tokenizer does not stop processing
    //!            of text if it meets such character
    //! @attention function isn't thread-safe
    //    in case of spacePreserve==false all whitespaces are replaced with space because
    //    browsers normalize whitespaces: "a \t\n\r b" -> "a b" if tag <pre></pre> isn't used
    //    this change fixes incorrect hyphenations without tag <pre>: "HTML-\nfile" is not "HTMLfile"
    //    browser show this text as: "HTML- file"
    //    in case of urlDecode==true firstly tokenizer tries to decode percent encoded text:
    //    "%D1%82%D0%B5%D0%BA%D1%81%D1%82" -> "текст" and then start tokenization.
    //    By default it's true.
    void Tokenize(const wchar16* text,
                  size_t len,
                  const TTokenizerOptions& opts);

    //! all other Tokenize() functions are for backward compatibility
    void Tokenize(const wchar16* text,
                  size_t len,
                  bool spacePreserve = false,
                  TLangMask langMask = TLangMask());

#ifndef CATBOOST_OPENSOURCE
    //! converts the text from yandex encoding to unicode and calls to the main tokenizing function
    void Tokenize(const char* text,
                  size_t len,
                  bool spacePreserve = false,
                  TLangMask langMask = TLangMask());
#endif

    //! just calls to the main tokenizing function
    void Tokenize(TWtringBuf text,
                  bool spacePreserve = false,
                  TLangMask langMask = TLangMask()) {
        Tokenize(text.begin(),
                 text.size(),
                 spacePreserve,
                 langMask);
    }

    //can point to text, Buffer or whatever
    //set by NlpParser
    //lifetime of data is min(lifetime(text), lifetime(tokenizer))
    const wchar16* GetTextStart() const {
        return TextStart;
    }
};

inline bool IsSpecialTokenizerSymbol(wchar32 ch) {
    return ch >= 128 && NUnicode::CharHasType(ch, (1ULL << Sm_MATH) | (1ULL << Sc_CURRENCY) | (1ULL << So_OTHER));
}

bool IsSpecialTokenizerSymbol(const TWtringBuf s);

inline bool IsAsciiEmojiPart(wchar32 ch) {
    return ch < 128 && !IsAlnum(ch);
}

bool IsAsciiEmojiPart(const TWtringBuf s);

template <class TCallback>
class TCallbackTokenHandler: public ITokenHandler {
    public:
        TCallbackTokenHandler(TCallback callback)
            : Callback(callback)
        {
        }

        virtual void OnToken(const TWideToken& token, size_t origleng, NLP_TYPE type) override {
            Callback(token, origleng, type);
        }
    private:
        TCallback Callback;
};

template <class TCallback>
TCallbackTokenHandler<TCallback> MakeCallbackTokenHandler(const TCallback& callback) {
    return TCallbackTokenHandler<TCallback>(callback);
}
