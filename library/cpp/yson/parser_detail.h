#pragma once

#include "detail.h"

namespace NYson {
    namespace NDetail {
        ////////////////////////////////////////////////////////////////////////////////

        template <class TConsumer, class TBlockStream, bool EnableLinePositionInfo>
        class TParser
           : public TLexerBase<TBlockStream, EnableLinePositionInfo> {
        private:
            using TBase = TLexerBase<TBlockStream, EnableLinePositionInfo>;
            TConsumer* Consumer;
            bool ConsumeUntilEof_;

        public:
            TParser(
                const TBlockStream& blockStream,
                TConsumer* consumer,
                bool consumeUntilEof,
                TMaybe<ui64> memoryLimit)
                    : TBase(blockStream, memoryLimit)
                    , Consumer(consumer)
                    , ConsumeUntilEof_(consumeUntilEof)
            {
            }

            void DoParse(EYsonType ysonType) {
                switch (ysonType) {
                    case ::NYson::EYsonType::Node:
                        ParseNode<true>();
                        break;

                    case ::NYson::EYsonType::ListFragment:
                        ParseListFragment<true>(EndSymbol);
                        break;

                    case ::NYson::EYsonType::MapFragment:
                        ParseMapFragment<true>(EndSymbol);
                        break;

                    default:
                        Y_ABORT("unreachable");
                }

                if (ConsumeUntilEof_) {
                    while (!(TBase::IsFinished() && TBase::IsEmpty())) {
                        if (TBase::template SkipSpaceAndGetChar<true>() != EndSymbol) {
                            ythrow TYsonException() << "Stray '" << (*TBase::Begin()) << "' found";
                        } else if (!TBase::IsEmpty()) {
                            TBase::Advance(1);
                        }
                    }
                }
            }

            bool DoParseListFragment(bool first) {
                bool ret = first ? first : ParseListSeparator<true>(EndSymbol);
                return ret && ParseListItem<true>(EndSymbol);
            }

            void ParseAttributes() {
                Consumer->OnBeginAttributes();
                ParseMapFragment(EndAttributesSymbol);
                TBase::SkipCharToken(EndAttributesSymbol);
                Consumer->OnEndAttributes();
            }

            void ParseMap() {
                Consumer->OnBeginMap();
                ParseMapFragment(EndMapSymbol);
                TBase::SkipCharToken(EndMapSymbol);
                Consumer->OnEndMap();
            }

            void ParseList() {
                Consumer->OnBeginList();
                ParseListFragment(EndListSymbol);
                TBase::SkipCharToken(EndListSymbol);
                Consumer->OnEndList();
            }

            template <bool AllowFinish>
            void ParseNode() {
                return ParseNode<AllowFinish>(TBase::SkipSpaceAndGetChar());
            }

            template <bool AllowFinish>
            void ParseNode(char ch) {
                if (ch == BeginAttributesSymbol) {
                    TBase::Advance(1);
                    ParseAttributes();
                    ch = TBase::SkipSpaceAndGetChar();
                }

                switch (ch) {
                    case BeginMapSymbol:
                        TBase::Advance(1);
                        ParseMap();
                        break;

                    case BeginListSymbol:
                        TBase::Advance(1);
                        ParseList();
                        break;

                    case '"': {
                        TBase::Advance(1);
                        TStringBuf value;
                        TBase::ReadQuotedString(&value);
                        Consumer->OnStringScalar(value);
                        break;
                    }
                    case StringMarker: {
                        TBase::Advance(1);
                        TStringBuf value;
                        TBase::ReadBinaryString(&value);
                        Consumer->OnStringScalar(value);
                        break;
                    }
                    case Int64Marker: {
                        TBase::Advance(1);
                        i64 value;
                        TBase::ReadBinaryInt64(&value);
                        Consumer->OnInt64Scalar(value);
                        break;
                    }
                    case Uint64Marker: {
                        TBase::Advance(1);
                        ui64 value;
                        TBase::ReadBinaryUint64(&value);
                        Consumer->OnUint64Scalar(value);
                        break;
                    }
                    case DoubleMarker: {
                        TBase::Advance(1);
                        double value;
                        TBase::ReadBinaryDouble(&value);
                        Consumer->OnDoubleScalar(value);
                        break;
                    }
                    case FalseMarker: {
                        TBase::Advance(1);
                        Consumer->OnBooleanScalar(false);
                        break;
                    }
                    case TrueMarker: {
                        TBase::Advance(1);
                        Consumer->OnBooleanScalar(true);
                        break;
                    }
                    case EntitySymbol:
                        TBase::Advance(1);
                        Consumer->OnEntity();
                        break;

                    default: {
                        if (isdigit((unsigned char)ch) || ch == '-' || ch == '+') { // case of '+' is handled in AfterPlus state
                            ReadNumeric<AllowFinish>();
                        } else if (isalpha((unsigned char)ch) || ch == '_') {
                            TStringBuf value;
                            TBase::template ReadUnquotedString<AllowFinish>(&value);
                            Consumer->OnStringScalar(value);
                        } else if (ch == '%') {
                            TBase::Advance(1);
                            ch = TBase::template GetChar<AllowFinish>();
                            if (ch == 't' || ch == 'f') {
                                Consumer->OnBooleanScalar(TBase::template ReadBoolean<AllowFinish>());
                            } else {
                                Consumer->OnDoubleScalar(TBase::template ReadNanOrInf<AllowFinish>());
                            }
                        } else {
                            ythrow TYsonException() << "Unexpected '" << ch << "' while parsing node";
                        }
                    }
                }
            }

            void ParseKey() {
                return ParseKey(TBase::SkipSpaceAndGetChar());
            }

            void ParseKey(char ch) {
                switch (ch) {
                    case '"': {
                        TBase::Advance(1);
                        TStringBuf value;
                        TBase::ReadQuotedString(&value);
                        Consumer->OnKeyedItem(value);
                        break;
                    }
                    case StringMarker: {
                        TBase::Advance(1);
                        TStringBuf value;
                        TBase::ReadBinaryString(&value);
                        Consumer->OnKeyedItem(value);
                        break;
                    }
                    default: {
                        if (isalpha(ch) || ch == '_') {
                            TStringBuf value;
                            TBase::ReadUnquotedString(&value);
                            Consumer->OnKeyedItem(value);
                        } else {
                            ythrow TYsonException() << "Unexpected '" << ch << "' while parsing key";
                        }
                    }
                }
            }

            template <bool AllowFinish>
            void ParseMapFragment(char endSymbol) {
                char ch = TBase::template SkipSpaceAndGetChar<AllowFinish>();
                while (ch != endSymbol) {
                    ParseKey(ch);
                    ch = TBase::template SkipSpaceAndGetChar<AllowFinish>();
                    if (ch == KeyValueSeparatorSymbol) {
                        TBase::Advance(1);
                    } else {
                        ythrow TYsonException() << "Expected '" << KeyValueSeparatorSymbol << "' but '" << ch << "' found";
                    }
                    ParseNode<AllowFinish>();
                    ch = TBase::template SkipSpaceAndGetChar<AllowFinish>();
                    if (ch == KeyedItemSeparatorSymbol) {
                        TBase::Advance(1);
                        ch = TBase::template SkipSpaceAndGetChar<AllowFinish>();
                    } else if (ch != endSymbol) {
                        ythrow TYsonException() << "Expected '" << KeyedItemSeparatorSymbol
                                                << "' or '\\0' ' but '" << ch << "' found";
                    }
                }
            }

            void ParseMapFragment(char endSymbol) {
                ParseMapFragment<false>(endSymbol);
            }

            template <bool AllowFinish>
            bool ParseListItem(char endSymbol) {
                char ch = TBase::template SkipSpaceAndGetChar<AllowFinish>();
                if (ch != endSymbol) {
                    Consumer->OnListItem();
                    ParseNode<AllowFinish>(ch);
                    return true;
                }
                return false;
            }

            template <bool AllowFinish>
            bool ParseListSeparator(char endSymbol) {
                char ch = TBase::template SkipSpaceAndGetChar<AllowFinish>();
                if (ch == ListItemSeparatorSymbol) {
                    TBase::Advance(1);
                    return true;
                } else if (ch != endSymbol) {
                    ythrow TYsonException() << "Expected '" << ListItemSeparatorSymbol
                                            << "' or '\\0' but '" << ch << "' found";
                }
                return false;
            }

            template <bool AllowFinish>
            void ParseListFragment(char endSymbol) {
                while (ParseListItem<AllowFinish>(endSymbol) && ParseListSeparator<AllowFinish>(endSymbol)) {
                }
            }

            void ParseListFragment(char endSymbol) {
                ParseListFragment<false>(endSymbol);
            }

            template <bool AllowFinish>
            void ReadNumeric() {
                TStringBuf valueBuffer;
                ENumericResult numericResult = TBase::template ReadNumeric<AllowFinish>(&valueBuffer);

                if (numericResult == ENumericResult::Double) {
                    double value;
                    try {
                        value = FromString<double>(valueBuffer);
                    } catch (yexception& e) {
                        // This exception is wrapped in parser.
                        ythrow TYsonException() << "Failed to parse double literal '" << valueBuffer << "'" << e;
                    }
                    Consumer->OnDoubleScalar(value);
                } else if (numericResult == ENumericResult::Int64) {
                    i64 value;
                    try {
                        value = FromString<i64>(valueBuffer);
                    } catch (yexception& e) {
                        // This exception is wrapped in parser.
                        ythrow TYsonException() << "Failed to parse int64 literal '" << valueBuffer << "'" << e;
                    }
                    Consumer->OnInt64Scalar(value);
                } else if (numericResult == ENumericResult::Uint64) {
                    ui64 value;
                    try {
                        value = FromString<ui64>(valueBuffer.SubStr(0, valueBuffer.size() - 1));
                    } catch (yexception& e) {
                        // This exception is wrapped in parser.
                        ythrow TYsonException() << "Failed to parse uint64 literal '" << valueBuffer << "'" << e;
                    }
                    Consumer->OnUint64Scalar(value);
                }
            }
        };

        ////////////////////////////////////////////////////////////////////////////////

    }

    template <class TConsumer, class TBlockStream>
    void ParseYsonStreamImpl(
        const TBlockStream& blockStream,
        NYT::NYson::IYsonConsumer* consumer,
        EYsonType parsingMode,
        bool enableLinePositionInfo,
        bool consumeUntilEof,
        TMaybe<ui64> memoryLimit) {
        if (enableLinePositionInfo) {
            using TImpl = NDetail::TParser<TConsumer, TBlockStream, true>;
            TImpl impl(blockStream, consumer, consumeUntilEof, memoryLimit);
            impl.DoParse(parsingMode);
        } else {
            using TImpl = NDetail::TParser<TConsumer, TBlockStream, false>;
            TImpl impl(blockStream, consumer, consumeUntilEof, memoryLimit);
            impl.DoParse(parsingMode);
        }
    }

    class TStatelessYsonParserImplBase {
    public:
        virtual void Parse(const TStringBuf& data, EYsonType type = ::NYson::EYsonType::Node) = 0;

        virtual ~TStatelessYsonParserImplBase() {
        }
    };

    template <class TConsumer, bool EnableLinePositionInfo>
    class TStatelessYsonParserImpl
       : public TStatelessYsonParserImplBase {
    private:
        using TParser = NDetail::TParser<TConsumer, TStringReader, EnableLinePositionInfo>;
        TParser Parser;

    public:
        TStatelessYsonParserImpl(TConsumer* consumer, TMaybe<ui64> memoryLimit)
            : Parser(TStringReader(), consumer, true, memoryLimit)
        {
        }

        void Parse(const TStringBuf& data, EYsonType type = ::NYson::EYsonType::Node) override {
            Parser.SetBuffer(data.begin(), data.end());
            Parser.DoParse(type);
        }
    };

    class TYsonListParserImplBase {
    public:
        virtual bool Parse() = 0;

        virtual ~TYsonListParserImplBase() {
        }
    };

    template <class TConsumer, class TBlockStream, bool EnableLinePositionInfo>
    class TYsonListParserImpl
       : public TYsonListParserImplBase {
    private:
        using TParser = NDetail::TParser<TConsumer, TBlockStream, EnableLinePositionInfo>;
        TParser Parser;
        bool First = true;

    public:
        TYsonListParserImpl(const TBlockStream& blockStream, TConsumer* consumer, TMaybe<ui64> memoryLimit)
            : Parser(blockStream, consumer, true, memoryLimit)
        {
        }

        bool Parse() override {
            bool ret = Parser.DoParseListFragment(First);
            First = false;
            return ret;
        }
    };

    ////////////////////////////////////////////////////////////////////////////////

} // namespace NYson
