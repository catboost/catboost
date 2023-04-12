#pragma once

class TLuaStateHolder;

namespace NJson {
    class TJsonValue;
}

namespace NLua {
    // Try to push TJsonValue to lua stack.
    // Lua stack state is undefined if there's not enough memory to grow stack appropriately
    // Exception is thrown in this case
    void PushJsonValue(TLuaStateHolder* state, const NJson::TJsonValue& json);
}
