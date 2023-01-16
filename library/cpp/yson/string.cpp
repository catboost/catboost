#include "string.h"
#include "parser.h"
#include "consumer.h"

namespace NYson {

////////////////////////////////////////////////////////////////////////////////

TYsonStringBuf::TYsonStringBuf()
{
    Type_ = EYsonType::Node; // fake
    Null_ = true;
}

TYsonStringBuf::TYsonStringBuf(const TString& data, EYsonType type)
    : TYsonStringBuf(TStringBuf(data), type)
{ }

TYsonStringBuf::TYsonStringBuf(TStringBuf data, EYsonType type)
    : Data_(data)
    , Type_(type)
    , Null_(false)
{ }

TYsonStringBuf::TYsonStringBuf(const char* data, EYsonType type)
    : TYsonStringBuf(TStringBuf(data), type)
{ }

TYsonStringBuf::operator bool() const
{
    return !Null_;
}

TStringBuf TYsonStringBuf::AsStringBuf() const
{
    Y_VERIFY(*this);
    return Data_;
}

EYsonType TYsonStringBuf::GetType() const
{
    Y_VERIFY(*this);
    return Type_;
}

////////////////////////////////////////////////////////////////////////////////

void Serialize(const TYsonStringBuf& yson, IYsonConsumer* consumer)
{
    consumer->OnRaw(yson);
}

TString ToString(const TYsonStringBuf& yson)
{
    return TString(yson.AsStringBuf());
}

////////////////////////////////////////////////////////////////////////////////

} // namespace NYson
