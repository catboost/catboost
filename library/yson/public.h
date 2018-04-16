#pragma once

#include <util/generic/yexception.h>

namespace NYT {
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

    class TYsonException
       : public yexception {};

    ////////////////////////////////////////////////////////////////////////////////

}
