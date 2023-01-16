#pragma once

#include "public.h"

#include <variant>

namespace NYson {

////////////////////////////////////////////////////////////////////////////////

//! Contains a sequence of bytes in YSON encoding annotated with EYsonType describing
//! the content. Could be null. Non-owning.
class TYsonStringBuf
{
public:
    //! Constructs a null instance.
    TYsonStringBuf();

    //! Constructs a non-null instance with given type and content.
    explicit TYsonStringBuf(
        const TString& data,
        EYsonType type = EYsonType::Node);

    //! Constructs a non-null instance with given type and content.
    explicit TYsonStringBuf(
        TStringBuf data,
        EYsonType type = EYsonType::Node);

    //! Constructs a non-null instance with given type and content
    //! (without this overload there is no way to construct TYsonStringBuf from
    //! string literal).
    explicit TYsonStringBuf(
        const char* data,
        EYsonType type = EYsonType::Node);

    //! Returns |true| if the instance is not null.
    explicit operator bool() const;

    //! Returns the underlying YSON bytes. The instance must be non-null.
    TStringBuf AsStringBuf() const;

    //! Returns type of YSON contained here. The instance must be non-null.
    EYsonType GetType() const;

protected:
    TStringBuf Data_;
    EYsonType Type_;
    bool Null_;
};

////////////////////////////////////////////////////////////////////////////////

void Serialize(const TYsonStringBuf& yson, IYsonConsumer* consumer);

bool operator == (const TYsonStringBuf& lhs, const TYsonStringBuf& rhs);
bool operator != (const TYsonStringBuf& lhs, const TYsonStringBuf& rhs);

TString ToString(const TYsonStringBuf& yson);

////////////////////////////////////////////////////////////////////////////////

} // namespace NYson

#define STRING_INL_H_
#include "string-inl.h"
#undef STRING_INL_H_
