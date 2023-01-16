#pragma once

#include "token.h"

namespace NYson {
    ////////////////////////////////////////////////////////////////////////////////

    const ETokenType BeginListToken = LeftBracket;
    const ETokenType EndListToken = RightBracket;

    const ETokenType BeginMapToken = LeftBrace;
    const ETokenType EndMapToken = RightBrace;

    const ETokenType BeginAttributesToken = LeftAngle;
    const ETokenType EndAttributesToken = RightAngle;

    const ETokenType ListItemSeparatorToken = Semicolon;
    const ETokenType KeyedItemSeparatorToken = Semicolon;
    const ETokenType KeyValueSeparatorToken = Equals;

    const ETokenType EntityToken = Hash;

    ////////////////////////////////////////////////////////////////////////////////

} // namespace NYson
