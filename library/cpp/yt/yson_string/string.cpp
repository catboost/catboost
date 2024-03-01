#include "string.h"

#include <library/cpp/yt/assert/assert.h>

#include <library/cpp/yt/misc/variant.h>

#include <library/cpp/yt/memory/new.h>

namespace NYT::NYson {

////////////////////////////////////////////////////////////////////////////////

TYsonStringBuf::TYsonStringBuf()
{
    Type_ = EYsonType::Node; // fake
    Null_ = true;
}

TYsonStringBuf::TYsonStringBuf(const TYsonString& ysonString)
{
    if (ysonString) {
        Data_ = ysonString.AsStringBuf();
        Type_ = ysonString.GetType();
        Null_ = false;
    } else {
        Type_ = EYsonType::Node; // fake
        Null_ = true;
    }
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
    YT_VERIFY(*this);
    return Data_;
}

EYsonType TYsonStringBuf::GetType() const
{
    YT_VERIFY(*this);
    return Type_;
}

////////////////////////////////////////////////////////////////////////////////

TYsonString::TYsonString()
{
    Begin_ = nullptr;
    Size_ = 0;
    Type_ = EYsonType::Node; // fake
}

TYsonString::TYsonString(const TYsonStringBuf& ysonStringBuf)
{
    if (ysonStringBuf) {
        auto data = ysonStringBuf.AsStringBuf();
        auto holder = NDetail::TYsonStringHolder::Allocate(data.length());
        ::memcpy(holder->GetData(), data.data(), data.length());
        Begin_ = holder->GetData();
        Size_ = data.Size();
        Type_ = ysonStringBuf.GetType();
        Payload_ = std::move(holder);
    } else {
        Begin_ = nullptr;
        Size_ = 0;
        Type_ = EYsonType::Node; // fake
    }
}

TYsonString::TYsonString(
    TStringBuf data,
    EYsonType type)
    : TYsonString(TYsonStringBuf(data, type))
{ }

#ifdef TSTRING_IS_STD_STRING
TYsonString::TYsonString(
    const TString& data,
    EYsonType type)
    : TYsonString(TYsonStringBuf(data, type))
{ }
#else
TYsonString::TYsonString(
    const TString& data,
    EYsonType type)
{
    // NOTE: CoW TString implementation is assumed
    // Moving the payload MUST NOT invalidate its internal pointers
    Payload_ = data;
    Begin_ = data.data();
    Size_ = data.length();
    Type_ = type;
}
#endif

TYsonString::TYsonString(
    const TSharedRef& data,
    EYsonType type)
{
    Payload_ = data.GetHolder();
    Begin_ = data.Begin();
    Size_ = data.Size();
    Type_ = type;
}

TYsonString::operator bool() const
{
    return !std::holds_alternative<TNullPayload>(Payload_);
}

EYsonType TYsonString::GetType() const
{
    YT_VERIFY(*this);
    return Type_;
}

TStringBuf TYsonString::AsStringBuf() const
{
    YT_VERIFY(*this);
    return TStringBuf(Begin_, Begin_ + Size_);
}

TString TYsonString::ToString() const
{
    return Visit(
        Payload_,
        [] (const TNullPayload&) -> TString {
            YT_ABORT();
        },
        [&] (const TSharedRangeHolderPtr&) {
            return TString(AsStringBuf());
        },
        [] (const TString& payload) {
            return payload;
        });
}

TSharedRef TYsonString::ToSharedRef() const
{
    return Visit(
        Payload_,
        [] (const TNullPayload&) -> TSharedRef {
            YT_ABORT();
        },
        [&] (const TSharedRangeHolderPtr& holder) {
            return TSharedRef(Begin_, Size_, holder);
        },
        [] (const TString& payload) {
            return TSharedRef::FromString(payload);
        });
}

size_t TYsonString::ComputeHash() const
{
    return THash<TStringBuf>()(TStringBuf(Begin_, Begin_ + Size_));
}

void TYsonString::Save(IOutputStream* s) const
{
    EYsonType type = Type_;
    if (*this) {
        ::SaveMany(s, type, ToSharedRef());
    } else {
        ::SaveMany(s, type, TString());
    }
}

void TYsonString::Load(IInputStream* s)
{
    EYsonType type;
    TString data;
    ::LoadMany(s, type, data);
    if (data) {
        *this = TYsonString(data, type);
    } else {
        *this = TYsonString();
    }
}

////////////////////////////////////////////////////////////////////////////////

TString ToString(const TYsonString& yson)
{
    return yson.ToString();
}

TString ToString(const TYsonStringBuf& yson)
{
    return TString(yson.AsStringBuf());
}

////////////////////////////////////////////////////////////////////////////////

} // namespace NYT::NYson
