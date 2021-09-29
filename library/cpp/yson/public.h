#pragma once

#include <library/cpp/ytalloc/core/misc/enum.h>
#include <util/generic/yexception.h>

namespace NYson {
    ////////////////////////////////////////////////////////////////////////////////

    //! The data format.
    DEFINE_ENUM(EYsonFormat,
        // Binary.
        // Most compact but not human-readable.
        (Binary)

        // Text.
        // Not so compact but human-readable.
        // Does not use indentation.
        // Uses escaping for non-text characters.
        (Text)

        // Text with indentation.
        // Extremely verbose but human-readable.
        // Uses escaping for non-text characters.
        (Pretty)
    );

    enum EYsonType {
        YT_NODE,
        YT_LIST_FRAGMENT,
        YT_MAP_FRAGMENT
    };

    struct IYsonConsumer;
    struct TYsonConsumerBase;

    class TYsonWriter;
    class TYsonParser;
    class TStatelessYsonParser;
    class TYsonListParser;

    class TYsonException
       : public yexception {};

    ////////////////////////////////////////////////////////////////////////////////

} // namespace NYson

namespace NYT {

using ::NYson::EYsonType;
using ::NYson::EYsonType::YT_NODE;
using ::NYson::EYsonType::YT_LIST_FRAGMENT;
using ::NYson::EYsonType::YT_MAP_FRAGMENT;

} // namespace NYT
