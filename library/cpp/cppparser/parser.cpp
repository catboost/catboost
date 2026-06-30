#include <util/generic/hash.h>
#include <util/string/ascii.h>
#include <util/string/cast.h>
#include <util/generic/hash_set.h>
#include <util/generic/yexception.h>

#include "parser.h"

//#define DEBUG_ME 1

TCppSaxParser::TText::TText()
    : Offset(0)
{
}

TCppSaxParser::TText::TText(ui64 offset)
    : Offset(offset)
{
}

TCppSaxParser::TText::TText(const TString& data, ui64 offset)
    : Data(data)
    , Offset(offset)
{
}

TCppSaxParser::TText::~TText() = default;

void TCppSaxParser::TText::Reset() noexcept {
    Offset += Data.length();
    Data.clear();
}

TCppSaxParser::TWorker::TWorker() noexcept = default;

TCppSaxParser::TWorker::~TWorker() = default;

class TCppSaxParser::TImpl {
    enum EState {
        Code,
        CommentBegin,
        String,
        Character,
        OneLineComment,
        MultiLineComment,
        MultiLineCommentEnd,
        Preprocessor
    };

public:
    typedef TCppSaxParser::TText TText;
    typedef TCppSaxParser::TWorker TWorker;

    inline TImpl(TWorker* worker)
        : State_(Code)
        , Worker_(worker)
        , SkipNext_(false)
        , Line_(0)
        , Column_(0)
    {
        Worker_->DoStart();
    }

    inline ~TImpl() = default;

    inline void Write(const void* data, size_t len) {
        ProcessInput((const char*)data, len);
    }

    inline void Finish() {
        if (!Text_.Data.empty()) {
            switch (State_) {
                case Code:
                    Worker_->DoCode(Text_);

                    break;

                case Preprocessor:
                    Worker_->DoPreprocessor(Text_);

                    break;

                case OneLineComment:
                    Worker_->DoOneLineComment(Text_);

                    break;

                default:
                    ThrowError();
            }
        }

        Worker_->DoEnd();
    }

private:
    inline void ProcessInput(const char* data, size_t len) {
        EState savedState = Code;
        while (len) {
            const char ch = *data;

            if (ch == '\n') {
                ++Line_;
                Column_ = 0;
            } else {
                ++Column_;
            }

#if DEBUG_ME
            Cerr << "char: " << ch << Endl;
            Cerr << "state before: " << (unsigned int)State_ << Endl;
#endif

        retry:
            switch (State_) {
                case Code: {
                    savedState = Code;
                    switch (ch) {
                        case '/':
                            State_ = CommentBegin;

                            break;

                        case '"':
                            Action(ch);
                            State_ = String;

                            break;

                        case '\'':
                            if (QuoteCharIsADigitSeparator()) {
                                Text_.Data += ch;
                                break;
                            }
                            Action(ch);
                            State_ = Character;

                            break;

                        case '#':
                            Action(ch);
                            State_ = Preprocessor;

                            break;

                        default:
                            Text_.Data += ch;

                            break;
                    }

                    break;
                }

                case CommentBegin: {
                    switch (ch) {
                        case '/':
                            State_ = savedState;
                            savedState = Code;
                            Action("//");
                            State_ = OneLineComment;

                            break;

                        case '*':
                            State_ = savedState;
                            Action("/*");
                            State_ = MultiLineComment;

                            break;

                        default:
                            Text_.Data += '/';
                            State_ = savedState;

                            goto retry;
                    }

                    break;
                }

                case OneLineComment: {
                    switch (ch) {
                        case '\n':
                            Action(ch);
                            State_ = Code;

                            break;

                        default:
                            Text_.Data += ch;

                            break;
                    }

                    break;
                }

                case MultiLineComment: {
                    switch (ch) {
                        case '*':
                            Text_.Data += ch;
                            State_ = MultiLineCommentEnd;

                            break;

                        case '\n':
                            Text_.Data += ch;
                            savedState = Code;

                            break;
                        default:
                            Text_.Data += ch;

                            break;
                    }

                    break;
                }

                case MultiLineCommentEnd: {
                    switch (ch) {
                        case '/':
                            Text_.Data += ch;
                            Action();
                            State_ = savedState;

                            break;

                        default:
                            State_ = MultiLineComment;

                            goto retry;
                    }

                    break;
                }

                case String: {
                    switch (ch) {
                        case '"':
                            Text_.Data += ch;

                            if (SkipNext_) {
                                SkipNext_ = false;
                            } else {
                                if (savedState == Code) {
                                    Action();
                                }
                                State_ = savedState;
                            }

                            break;

                        case '\\':
                            Text_.Data += ch;
                            SkipNext_ = !SkipNext_;

                            break;

                        default:
                            Text_.Data += ch;
                            SkipNext_ = false;

                            break;
                    }

                    break;
                }

                case Character: {
                    switch (ch) {
                        case '\'':
                            Text_.Data += ch;

                            if (SkipNext_) {
                                SkipNext_ = false;
                            } else {
                                if (savedState == Code) {
                                    Action();
                                }
                                State_ = savedState;
                            }

                            break;

                        case '\\':
                            Text_.Data += ch;
                            SkipNext_ = !SkipNext_;

                            break;

                        default:
                            Text_.Data += ch;
                            SkipNext_ = false;

                            break;
                    }

                    break;
                }

                case Preprocessor: {
                    savedState = Preprocessor;
                    switch (ch) {
                        case '/':
                            State_ = CommentBegin;

                            break;

                        case '\'':
                            Text_.Data += ch;
                            State_ = Character;

                            break;

                        case '"':
                            Text_.Data += ch;
                            State_ = String;

                            break;
                        case '\n':
                            Text_.Data += ch;

                            if (SkipNext_) {
                                SkipNext_ = false;
                            } else {
                                Action();
                                savedState = Code;
                                State_ = Code;
                            }

                            break;

                        case '\\':
                            Text_.Data += ch;
                            SkipNext_ = true;

                            break;

                        default:
                            Text_.Data += ch;
                            SkipNext_ = false;

                            break;
                    }

                    break;
                }

                default:
                    ThrowError();
            }

#if DEBUG_ME
            Cerr << "state after: " << (unsigned int)State_ << Endl;
#endif

            ++data;
            --len;
        }
    }

    // digit separator in integral literal (ex. 73'709'550'592)
    bool QuoteCharIsADigitSeparator() const {
        const TStringBuf data = Text_.Data;
        if (data.empty()) {
            return false;
        }
        if (!IsAsciiHex(data.back())) {
            return false;
        }
        // check for char literal prefix (ex. `u8'$'`)
        static constexpr TStringBuf literalPrefixes[] {
            "u8",
            "u",
            "U",
            "L",
        };
        for (const TStringBuf& literalPrefix : literalPrefixes) {
            if (TStringBuf prev; data.BeforeSuffix(literalPrefix, prev)) {
                if (!prev.empty() && (IsAsciiAlnum(prev.back()) || prev.back() == '_' || prev.back() == '$')) {
                    // some macro name ends with an `u8` sequence
                    continue;
                }
                // it is a prefixed character literal
                return false;
            }
        }
        return true;
    }

    inline void Action(char ch) {
        Action();
        Text_.Data += ch;
    }

    inline void Action(const char* st) {
        Action();
        Text_.Data += st;
    }

    inline void Action() {
        switch (State_) {
            case Code:
                Worker_->DoCode(Text_);

                break;

            case OneLineComment:
                Worker_->DoOneLineComment(Text_);

                break;

            case MultiLineCommentEnd:
                Worker_->DoMultiLineComment(Text_);

                break;

            case Preprocessor:
                Worker_->DoPreprocessor(Text_);

                break;

            case String:
                Worker_->DoString(Text_);

                break;

            case Character:
                Worker_->DoCharacter(Text_);

                break;

            default:
                ThrowError();
        }

        Text_.Reset();
    }

    inline void ThrowError() const {
        ythrow yexception() << "can not parse source(line = " << (unsigned)Line_ + 1 << ", column = " << (unsigned)Column_ + 1 << ")";
    }

private:
    EState State_;
    TWorker* Worker_;
    TText Text_;
    bool SkipNext_;
    ui64 Line_;
    ui64 Column_;
};

TCppSaxParser::TCppSaxParser(TWorker* worker)
    : Impl_(new TImpl(worker))
{
}

TCppSaxParser::~TCppSaxParser() = default;

void TCppSaxParser::DoWrite(const void* data, size_t len) {
    Impl_->Write(data, len);
}

void TCppSaxParser::DoFinish() {
    Impl_->Finish();
}

TCppSimpleSax::TCppSimpleSax() noexcept {
}

TCppSimpleSax::~TCppSimpleSax() = default;

void TCppSimpleSax::DoCode(const TText& text) {
    static const char char_types[] = {
        2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 0, 2, 2, 0, 2, 2,
        2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
        0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2,
        2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 1,
        2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2,
        2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
        2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
        2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
        2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
        2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
        2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
        2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
        2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2};

    static const char CWHITESPACE = 0;
    static const char CIDENTIFIER = 1;
    static const char CSYNTAX = 2;

    enum EState {
        WhiteSpace = CWHITESPACE,
        Identifier = CIDENTIFIER,
        Syntax = CSYNTAX
    };

    EState state = Identifier;
    TText cur(text.Offset);

    for (const auto& it : text.Data) {
        const unsigned char ch = *(const unsigned char*)(&it);
        const char type = char_types[ch];

        switch (state) {
            case Identifier: {
                switch (type) {
                    case CIDENTIFIER:
                        cur.Data += ch;

                        break;

                    default:
                        if (!cur.Data.empty()) {
                            DoIdentifier(cur);
                        }

                        cur.Reset();
                        cur.Data += ch;
                        state = (EState)type;

                        break;
                }

                break;
            }

            case WhiteSpace: {
                switch (type) {
                    case CWHITESPACE:
                        cur.Data += ch;

                        break;

                    default:
                        DoWhiteSpace(cur);
                        cur.Reset();
                        cur.Data += ch;
                        state = (EState)type;

                        break;
                }

                break;
            }

            case Syntax: {
                switch (type) {
                    case CSYNTAX:
                        cur.Data += ch;

                        break;

                    default:
                        DoSyntax(cur);
                        cur.Reset();
                        cur.Data += ch;
                        state = (EState)type;

                        break;
                }

                break;
            }
        }
    }

    if (!cur.Data.empty()) {
        switch (state) {
            case Identifier:
                DoIdentifier(cur);

                break;

            case WhiteSpace:
                DoWhiteSpace(cur);

                break;

            case Syntax:
                DoSyntax(cur);

                break;
        }
    }
}

class TCppFullSax::TImpl {
    typedef THashSet<TString> TKeyWords;

    class TRegExp {
    public:
        inline TRegExp(const char*) {
        }

        inline bool Match(const TString& /*s*/) const noexcept {
            return false;
        }
    };

public:
    inline TImpl()
        : OctNumber_("^[+-]?0[0-7]+$")
        , HexNumber_("^[+-]?0x[0-9A-Fa-f]+$")
        , DecNumber_("^[+-]?[0-9]+$")
        , FltNumber_("^[+-]?[0-9]*\\.[0-9]*$")
    {
        AddKeyword("extern");
        AddKeyword("static");
        AddKeyword("inline");
        AddKeyword("volatile");
        AddKeyword("asm");
        AddKeyword("const");
        AddKeyword("mutable");
        AddKeyword("char");
        AddKeyword("signed");
        AddKeyword("unsigned");
        AddKeyword("int");
        AddKeyword("short");
        AddKeyword("long");
        AddKeyword("double");
        AddKeyword("float");
        AddKeyword("bool");
        AddKeyword("class");
        AddKeyword("struct");
        AddKeyword("union");
        AddKeyword("void");
        AddKeyword("auto");
        AddKeyword("throw");
        AddKeyword("try");
        AddKeyword("catch");
        AddKeyword("for");
        AddKeyword("do");
        AddKeyword("if");
        AddKeyword("else");
        AddKeyword("while");
        AddKeyword("switch");
        AddKeyword("case");
        AddKeyword("default");
        AddKeyword("goto");
        AddKeyword("break");
        AddKeyword("continue");
        AddKeyword("virtual");
        AddKeyword("template");
        AddKeyword("typename");
        AddKeyword("enum");
        AddKeyword("public");
        AddKeyword("private");
        AddKeyword("protected");
        AddKeyword("using");
        AddKeyword("namespace");
        AddKeyword("typedef");
        AddKeyword("true");
        AddKeyword("false");
        AddKeyword("return");
        AddKeyword("new");
        AddKeyword("delete");
        AddKeyword("operator");
        AddKeyword("friend");
        AddKeyword("this");
    }

    inline ~TImpl() = default;

    inline void AddKeyword(const TString& keyword) {
        KeyWords_.insert(keyword);
    }

    inline bool IsKeyword(const TString& s) {
        return KeyWords_.find(s) != KeyWords_.end();
    }

    inline bool IsOctNumber(const TString& s) {
        return OctNumber_.Match(s);
    }

    inline bool IsHexNumber(const TString& s) {
        return HexNumber_.Match(s);
    }

    inline bool IsDecNumber(const TString& s) {
        return DecNumber_.Match(s);
    }

    inline bool IsFloatNumber(const TString& s) {
        return FltNumber_.Match(s);
    }

private:
    const TRegExp OctNumber_;
    const TRegExp HexNumber_;
    const TRegExp DecNumber_;
    const TRegExp FltNumber_;
    TKeyWords KeyWords_;
};

TCppFullSax::TCppFullSax()
    : Impl_(new TImpl())
{
}

TCppFullSax::~TCppFullSax() = default;

void TCppFullSax::AddKeyword(const TString& keyword) {
    Impl_->AddKeyword(keyword);
}

void TCppFullSax::DoIdentifier(const TText& text) {
    if (Impl_->IsKeyword(text.Data)) {
        DoKeyword(text);
    } else if (Impl_->IsOctNumber(text.Data)) {
        DoOctNumber(text);
    } else if (Impl_->IsHexNumber(text.Data)) {
        DoHexNumber(text);
    } else if (Impl_->IsDecNumber(text.Data)) {
        DoDecNumber(text);
    } else if (Impl_->IsFloatNumber(text.Data)) {
        DoFloatNumber(text);
    } else {
        DoName(text);
    }
}

void TCppFullSax::DoEnd() {
}

void TCppFullSax::DoStart() {
}

void TCppFullSax::DoString(const TText&) {
}

void TCppFullSax::DoCharacter(const TText&) {
}

void TCppFullSax::DoWhiteSpace(const TText&) {
}

void TCppFullSax::DoKeyword(const TText&) {
}

void TCppFullSax::DoName(const TText&) {
}

void TCppFullSax::DoOctNumber(const TText&) {
}

void TCppFullSax::DoHexNumber(const TText&) {
}

void TCppFullSax::DoDecNumber(const TText&) {
}

void TCppFullSax::DoFloatNumber(const TText&) {
}

void TCppFullSax::DoSyntax(const TText&) {
}

void TCppFullSax::DoOneLineComment(const TText&) {
}

void TCppFullSax::DoMultiLineComment(const TText&) {
}

void TCppFullSax::DoPreprocessor(const TText&) {
}
