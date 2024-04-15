#pragma once

#include "public.h"

#include <util/generic/maybe.h>
#include <util/generic/ptr.h>

class IInputStream;

namespace NYT::NYson {
struct IYsonConsumer;
} // namespace NYT::NYson

namespace NYson {
    ////////////////////////////////////////////////////////////////////////////////

    class TYsonParser {
    public:
        TYsonParser(
            NYT::NYson::IYsonConsumer* consumer,
            IInputStream* stream,
            EYsonType type = ::NYson::EYsonType::Node,
            bool enableLinePositionInfo = false,
            ui64 bufferSizeLimit = 64 << 10,
            bool consumeUntilEof = true,
            TMaybe<ui64> memoryLimit = Nothing());

        ~TYsonParser();

        void Parse();

    private:
        class TImpl;
        THolder<TImpl> Impl;
    };

    ////////////////////////////////////////////////////////////////////////////////

    class TStatelessYsonParser {
    public:
        TStatelessYsonParser(
            NYT::NYson::IYsonConsumer* consumer,
            bool enableLinePositionInfo = false,
            TMaybe<ui64> memoryLimit = Nothing());

        ~TStatelessYsonParser();

        void Parse(const TStringBuf& data, EYsonType type = ::NYson::EYsonType::Node);

    private:
        class TImpl;
        THolder<TImpl> Impl;
    };

    ////////////////////////////////////////////////////////////////////////////////

    class TYsonListParser {
    public:
        TYsonListParser(
            NYT::NYson::IYsonConsumer* consumer,
            IInputStream* stream,
            bool enableLinePositionInfo = false,
            TMaybe<ui64> memoryLimit = Nothing());

        ~TYsonListParser();

        bool Parse(); // Returns false, if there is no more list items

    private:
        class TImpl;
        THolder<TImpl> Impl;
    };

    ////////////////////////////////////////////////////////////////////////////////

    void ParseYsonStringBuffer(
        const TStringBuf& buffer,
        NYT::NYson::IYsonConsumer* consumer,
        EYsonType type = ::NYson::EYsonType::Node,
        bool enableLinePositionInfo = false,
        TMaybe<ui64> memoryLimit = Nothing());

    ////////////////////////////////////////////////////////////////////////////////

} // namespace NYson
