#include "json.h"
#include "wrapper.h"

#include <library/cpp/json/json_value.h>

using namespace NJson;

void NLua::PushJsonValue(TLuaStateHolder* state, const TJsonValue& json) {
    // each recursive call will explicitly push only a single element to stack relying on subcalls to reserve stack space for themselves
    // I.e. for a map {"a": "b"} the first call will ensure stack space for create_table, then call PushJsonValue for the string,
    // this PushJsonValue will ensure stack space for the string. Thus only a single ensure_stack at the start of the function is enough.
    state->ensure_stack(1);
    switch (json.GetType()) {
        case JSON_UNDEFINED:
            ythrow yexception() << "cannot push undefined json value";

        case JSON_NULL:
            state->push_nil();
            break;

        case JSON_BOOLEAN:
            state->push_bool(json.GetBoolean());
            break;

        case JSON_INTEGER:
            state->push_number(json.GetInteger());
            break;

        case JSON_UINTEGER:
            state->push_number(json.GetUInteger());
            break;

        case JSON_DOUBLE:
            state->push_number(json.GetDouble());
            break;

        case JSON_STRING:
            state->push_string(json.GetString());
            break;

        case JSON_MAP:
            state->create_table();
            for (const auto& pair : json.GetMap()) {
                PushJsonValue(state, pair.second); // Recursive call tests for stack space on its own
                state->set_field(-2, pair.first.data());
            }
            break;

        case JSON_ARRAY: {
            state->create_table();
            int index = 1; // lua arrays start from 1
            for (const auto& element : json.GetArray()) {
                PushJsonValue(state, element); // Recursive call tests for stack space on its own, no need to double check
                state->rawseti(-2, index++);
            }
            break;
        }

        default:
            ythrow yexception() << "Unexpected json value type";
    }
}
