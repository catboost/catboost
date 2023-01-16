#pragma once

#include "public.h"

#include <util/generic/strbuf.h>

namespace NYson {
    ////////////////////////////////////////////////////////////////////////////////

    enum ETokenType {
        EndOfStream,

        String,
        Int64,
        Uint64,
        Double,
        Boolean,

        // Special values:
        // YSON
        Semicolon,    // ;
        Equals,       // =
        Hash,         // #
        LeftBracket,  // [
        RightBracket, // ]
        LeftBrace,    // {
        RightBrace,   // }
        LeftAngle,    // <
        RightAngle,   // >

        // Table ranges
        LeftParenthesis,  // (
        RightParenthesis, // )
        Plus,             // +
        Colon,            // :
        Comma,            // ,
    };

    ////////////////////////////////////////////////////////////////////////////////

    ETokenType CharToTokenType(char ch);
    char TokenTypeToChar(ETokenType type);
    TString TokenTypeToString(ETokenType type);

    ////////////////////////////////////////////////////////////////////////////////

    class TLexerImpl;

    ////////////////////////////////////////////////////////////////////////////////

    class TToken {
    public:
        static const TToken EndOfStream;

        TToken();
        TToken(ETokenType type);
        explicit TToken(const TStringBuf& stringValue);
        explicit TToken(i64 int64Value);
        explicit TToken(ui64 int64Value);
        explicit TToken(double doubleValue);
        explicit TToken(bool booleanValue);

        ETokenType GetType() const {
            return Type_;
        }

        bool IsEmpty() const;
        const TStringBuf& GetStringValue() const;
        i64 GetInt64Value() const;
        ui64 GetUint64Value() const;
        double GetDoubleValue() const;
        bool GetBooleanValue() const;

        void CheckType(ETokenType expectedType) const;
        void Reset();

    private:
        friend class TLexerImpl;

        ETokenType Type_;

        TStringBuf StringValue;
        i64 Int64Value;
        ui64 Uint64Value;
        double DoubleValue;
        bool BooleanValue;
    };

    TString ToString(const TToken& token);

    ////////////////////////////////////////////////////////////////////////////////

} // namespace NYson
