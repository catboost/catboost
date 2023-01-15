#include "node_io.h"

#include "node_builder.h"
#include "node_visitor.h"

#include <library/cpp/yson/json_writer.h>
#include <library/cpp/yson/parser.h>
#include <library/cpp/yson/writer.h>
#include <library/cpp/yson/yson2json_adapter.h>

#include <library/cpp/json/json_reader.h>
#include <library/cpp/json/json_value.h>

#include <util/stream/input.h>
#include <util/stream/output.h>
#include <util/stream/str.h>
#include <util/stream/mem.h>

namespace NYT {

static void WalkJsonTree(const NJson::TJsonValue& jsonValue, NJson::TJsonCallbacks* callbacks)
{
    using namespace NJson;
    switch (jsonValue.GetType()) {
        case JSON_NULL:
            callbacks->OnNull();
            return;
        case JSON_BOOLEAN:
            callbacks->OnBoolean(jsonValue.GetBoolean());
            return;
        case JSON_INTEGER:
            callbacks->OnInteger(jsonValue.GetInteger());
            return;
        case JSON_UINTEGER:
            callbacks->OnUInteger(jsonValue.GetUInteger());
            return;
        case JSON_DOUBLE:
            callbacks->OnDouble(jsonValue.GetDouble());
            return;
        case JSON_STRING:
            callbacks->OnString(jsonValue.GetString());
            return;
        case JSON_MAP:
            {
                callbacks->OnOpenMap();
                for (const auto& item : jsonValue.GetMap()) {
                    callbacks->OnMapKey(item.first);
                    WalkJsonTree(item.second, callbacks);
                }
                callbacks->OnCloseMap();
            }
            return;
        case JSON_ARRAY:
            {
                callbacks->OnOpenArray();
                for (const auto& item : jsonValue.GetArray()) {
                    WalkJsonTree(item, callbacks);
                }
                callbacks->OnCloseArray();
            }
            return;
        case JSON_UNDEFINED:
            ythrow yexception() << "cannot consume undefined json value";
            return;
    }
    Y_UNREACHABLE();
}

static TNode CreateEmptyNodeByType(EYsonType type)
{
    TNode result;
    switch (type) {
        case YT_LIST_FRAGMENT:
            result = TNode::CreateList();
            break;
        case YT_MAP_FRAGMENT:
            result = TNode::CreateMap();
            break;
        default:
            break;
    }
    return result;
}

TNode NodeFromYsonString(const TStringBuf input, EYsonType type)
{
    TMemoryInput stream(input);
    return NodeFromYsonStream(&stream, type);
}

TString NodeToYsonString(const TNode& node, EYsonFormat format)
{
    TStringStream stream;
    NodeToYsonStream(node, &stream, format);
    return stream.Str();
}

TString NodeToCanonicalYsonString(const TNode& node, EYsonFormat format)
{
    TStringStream stream;
    NodeToCanonicalYsonStream(node, &stream, format);
    return stream.Str();
}

TNode NodeFromYsonStream(IInputStream* input, EYsonType type)
{
    TNode result = CreateEmptyNodeByType(type);

    TNodeBuilder builder(&result);
    TYsonParser parser(&builder, input, type);
    parser.Parse();
    return result;
}

void NodeToYsonStream(const TNode& node, IOutputStream* output, EYsonFormat format)
{
    TYsonWriter writer(output, format);
    TNodeVisitor visitor(&writer);
    visitor.Visit(node);
}

void NodeToCanonicalYsonStream(const TNode& node, IOutputStream* output, EYsonFormat format)
{
    TYsonWriter writer(output, format);
    TNodeVisitor visitor(&writer, /*sortMapKeys*/ true);
    visitor.Visit(node);
}

TNode NodeFromJsonString(const TStringBuf input)
{
    TMemoryInput stream(input);

    TNode result;

    TNodeBuilder builder(&result);
    TYson2JsonCallbacksAdapter callbacks(&builder, /*throwException*/ true);
    NJson::TJsonReaderConfig config;
    config.DontValidateUtf8 = true;
    NJson::ReadJson(&stream, &config, &callbacks);
    return result;
}

TNode NodeFromJsonValue(const NJson::TJsonValue& input)
{
    TNode result;
    TNodeBuilder builder(&result);
    TYson2JsonCallbacksAdapter callbacks(&builder, /*throwException*/ true);
    WalkJsonTree(input, &callbacks);
    return result;
}

////////////////////////////////////////////////////////////////////////////////

} // namespace NYT
