#pragma once

#include <library/cpp/yt/misc/enum.h>
#include <util/generic/yexception.h>

#include <library/cpp/yt/yson_string/public.h>
#include <library/cpp/yt/yson/public.h>

namespace NYson {

    ////////////////////////////////////////////////////////////////////////////////

    using NYT::NYson::EYsonFormat;
    using NYT::NYson::EYsonType;

    class TYsonStringBuf;

    struct TYsonConsumerBase;

    class TYsonWriter;
    class TYsonParser;
    class TStatelessYsonParser;
    class TYsonListParser;

    class TYsonException
       : public yexception {};

    ////////////////////////////////////////////////////////////////////////////////

} // namespace NYson
