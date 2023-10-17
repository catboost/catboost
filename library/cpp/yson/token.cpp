#include "token.h"

#include <util/string/vector.h>
#include <util/string/printf.h>

namespace NYson {
    ////////////////////////////////////////////////////////////////////////////////

    ETokenType CharToTokenType(char ch) {
        switch (ch) {
            case ';':
                return ETokenType::Semicolon;
            case '=':
                return ETokenType::Equals;
            case '{':
                return ETokenType::LeftBrace;
            case '}':
                return ETokenType::RightBrace;
            case '#':
                return ETokenType::Hash;
            case '[':
                return ETokenType::LeftBracket;
            case ']':
                return ETokenType::RightBracket;
            case '<':
                return ETokenType::LeftAngle;
            case '>':
                return ETokenType::RightAngle;
            case '(':
                return ETokenType::LeftParenthesis;
            case ')':
                return ETokenType::RightParenthesis;
            case '+':
                return ETokenType::Plus;
            case ':':
                return ETokenType::Colon;
            case ',':
                return ETokenType::Comma;
            default:
                return ETokenType::EndOfStream;
        }
    }

    char TokenTypeToChar(ETokenType type) {
        switch (type) {
            case ETokenType::Semicolon:
                return ';';
            case ETokenType::Equals:
                return '=';
            case ETokenType::Hash:
                return '#';
            case ETokenType::LeftBracket:
                return '[';
            case ETokenType::RightBracket:
                return ']';
            case ETokenType::LeftBrace:
                return '{';
            case ETokenType::RightBrace:
                return '}';
            case ETokenType::LeftAngle:
                return '<';
            case ETokenType::RightAngle:
                return '>';
            case ETokenType::LeftParenthesis:
                return '(';
            case ETokenType::RightParenthesis:
                return ')';
            case ETokenType::Plus:
                return '+';
            case ETokenType::Colon:
                return ':';
            case ETokenType::Comma:
                return ',';
            default:
                Y_ABORT("unreachable");
        }
    }

    TString TokenTypeToString(ETokenType type) {
        return TString(1, TokenTypeToChar(type));
    }

    ////////////////////////////////////////////////////////////////////////////////

    const TToken TToken::EndOfStream;

    TToken::TToken()
        : Type_(ETokenType::EndOfStream)
        , Int64Value(0)
        , Uint64Value(0)
        , DoubleValue(0.0)
        , BooleanValue(false)
    {
    }

    TToken::TToken(ETokenType type)
        : Type_(type)
        , Int64Value(0)
        , Uint64Value(0)
        , DoubleValue(0.0)
        , BooleanValue(false)
    {
        switch (type) {
            case ETokenType::String:
            case ETokenType::Int64:
            case ETokenType::Uint64:
            case ETokenType::Double:
            case ETokenType::Boolean:
                Y_ABORT("unreachable");
            default:
                break;
        }
    }

    TToken::TToken(const TStringBuf& stringValue)
        : Type_(ETokenType::String)
        , StringValue(stringValue)
        , Int64Value(0)
        , Uint64Value(0)
        , DoubleValue(0.0)
        , BooleanValue(false)
    {
    }

    TToken::TToken(i64 int64Value)
        : Type_(ETokenType::Int64)
        , Int64Value(int64Value)
        , Uint64Value(0)
        , DoubleValue(0.0)
    {
    }

    TToken::TToken(ui64 uint64Value)
        : Type_(ETokenType::Uint64)
        , Int64Value(0)
        , Uint64Value(uint64Value)
        , DoubleValue(0.0)
        , BooleanValue(false)
    {
    }

    TToken::TToken(double doubleValue)
        : Type_(ETokenType::Double)
        , Int64Value(0)
        , Uint64Value(0)
        , DoubleValue(doubleValue)
        , BooleanValue(false)
    {
    }

    TToken::TToken(bool booleanValue)
        : Type_(ETokenType::Boolean)
        , Int64Value(0)
        , DoubleValue(0.0)
        , BooleanValue(booleanValue)
    {
    }

    bool TToken::IsEmpty() const {
        return Type_ == ETokenType::EndOfStream;
    }

    const TStringBuf& TToken::GetStringValue() const {
        CheckType(ETokenType::String);
        return StringValue;
    }

    i64 TToken::GetInt64Value() const {
        CheckType(ETokenType::Int64);
        return Int64Value;
    }

    ui64 TToken::GetUint64Value() const {
        CheckType(ETokenType::Uint64);
        return Uint64Value;
    }

    double TToken::GetDoubleValue() const {
        CheckType(ETokenType::Double);
        return DoubleValue;
    }

    bool TToken::GetBooleanValue() const {
        CheckType(ETokenType::Boolean);
        return BooleanValue;
    }

    void TToken::CheckType(ETokenType expectedType) const {
        if (Type_ != expectedType) {
            if (Type_ == ETokenType::EndOfStream) {
                ythrow TYsonException() << "Unexpected end of stream (ExpectedType: " << TokenTypeToString(expectedType) << ")";
            } else {
                ythrow TYsonException() << "Unexpected token (Token: '" << ToString(*this)
                                        << "', Type: " << TokenTypeToString(Type_)
                                        << ", ExpectedType: " << TokenTypeToString(expectedType) << ")";
            }
        }
    }

    void TToken::Reset() {
        Type_ = ETokenType::EndOfStream;
        Int64Value = 0;
        Uint64Value = 0;
        DoubleValue = 0.0;
        StringValue = TStringBuf();
        BooleanValue = false;
    }

    TString ToString(const TToken& token) {
        switch (token.GetType()) {
            case ETokenType::EndOfStream:
                return TString();

            case ETokenType::String:
                return TString(token.GetStringValue());

            case ETokenType::Int64:
                return ::ToString(token.GetInt64Value());

            case ETokenType::Uint64:
                return ::ToString(token.GetUint64Value());

            case ETokenType::Double:
                return ::ToString(token.GetDoubleValue());

            case ETokenType::Boolean:
                return token.GetBooleanValue() ? "true" : "false";

            default:
                return TokenTypeToString(token.GetType());
        }
    }

    ////////////////////////////////////////////////////////////////////////////////

} // namespace NYson
