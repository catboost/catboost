#pragma once

#include <util/generic/yexception.h>

namespace NYson {
    ////////////////////////////////////////////////////////////////////////////////

    //! The data format.
    enum EYsonFormat {
        // Binary.
        // Most compact but not human-readable.
        YF_BINARY,

        // Text.
        // Not so compact but human-readable.
        // Does not use indentation.
        // Uses escaping for non-text characters.
        YF_TEXT,

        // Text with indentation.
        // Extremely verbose but human-readable.
        // Uses escaping for non-text characters.
        YF_PRETTY
    };

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

// Temporary for backward compatibility
using ::NYson::EYsonFormat;
using ::NYson::EYsonFormat::YF_BINARY;
using ::NYson::EYsonFormat::YF_TEXT;
using ::NYson::EYsonFormat::YF_PRETTY;

using ::NYson::EYsonType;
using ::NYson::EYsonType::YT_NODE;
using ::NYson::EYsonType::YT_LIST_FRAGMENT;
using ::NYson::EYsonType::YT_MAP_FRAGMENT;

using ::NYson::IYsonConsumer;
using ::NYson::TYsonConsumerBase;

using ::NYson::TYsonWriter;

using ::NYson::TYsonListParser;

} // namespace NYT
