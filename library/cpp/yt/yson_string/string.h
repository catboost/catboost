#pragma once

#include "public.h"

#include <library/cpp/yt/memory/ref.h>

#include <variant>

namespace NYT::NYson {

////////////////////////////////////////////////////////////////////////////////

//! Contains a sequence of bytes in YSON encoding annotated with EYsonType describing
//! the content. Could be null. Non-owning.
class TYsonStringBuf
{
public:
    //! Constructs a null instance.
    TYsonStringBuf();

    //! Constructs an instance from TYsonString.
    TYsonStringBuf(const TYsonString& ysonString);

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

//! An owning version of TYsonStringBuf.
/*!
 *  Internally captures the data either via TString or a polymorphic ref-counted holder.
 */
class TYsonString
{
public:
    //! Constructs a null instance.
    TYsonString();

    //! Constructs an instance from TYsonStringBuf.
    //! Copies the data into a ref-counted payload.
    explicit TYsonString(const TYsonStringBuf& ysonStringBuf);

    //! Constructs an instance from TStringBuf.
    //! Copies the data into a ref-counted payload.
    explicit TYsonString(
        TStringBuf data,
        EYsonType type = EYsonType::Node);

    //! Constructs an instance from TString.
    //! Zero-copy for CoW TString: retains the reference to TString in payload.
    explicit TYsonString(
        const TString& data,
        EYsonType type = EYsonType::Node);

    //! Constructs an instance from TSharedRef.
    //! Zero-copy; retains the reference to TSharedRef holder in payload.
    explicit TYsonString(
        const TSharedRef& ref,
        EYsonType type = EYsonType::Node);

    //! Returns |true| if the instance is not null.
    explicit operator bool() const;

    //! Returns type of YSON contained here. The instance must be non-null.
    EYsonType GetType() const;

    //! Returns the non-owning data. The instance must be non-null.
    TStringBuf AsStringBuf() const;

    //! Returns the data represented by TString. The instance must be non-null.
    //! Copies the data in case the payload is not TString.
    TString ToString() const;

    //! Returns the data represented by TSharedRef. The instance must be non-null.
    //! The data is never copied.
    TSharedRef ToSharedRef() const;

    //! Computes the hash code.
    size_t ComputeHash() const;

    //! Allow to serialize/deserialize using the ::Save ::Load functions. See util/ysaveload.h.
    void Save(IOutputStream* s) const;
    void Load(IInputStream* s);

private:
    struct TNullPayload
    { };

    std::variant<TNullPayload, TSharedRangeHolderPtr, TString> Payload_;

    const char* Begin_;
    ui64 Size_ : 56;
    EYsonType Type_ : 8;
};

////////////////////////////////////////////////////////////////////////////////

bool operator == (const TYsonString& lhs, const TYsonString& rhs);
bool operator == (const TYsonString& lhs, const TYsonStringBuf& rhs);
bool operator == (const TYsonStringBuf& lhs, const TYsonString& rhs);
bool operator == (const TYsonStringBuf& lhs, const TYsonStringBuf& rhs);

bool operator != (const TYsonString& lhs, const TYsonString& rhs);
bool operator != (const TYsonString& lhs, const TYsonStringBuf& rhs);
bool operator != (const TYsonStringBuf& lhs, const TYsonString& rhs);
bool operator != (const TYsonStringBuf& lhs, const TYsonStringBuf& rhs);

TString ToString(const TYsonString& yson);
TString ToString(const TYsonStringBuf& yson);

////////////////////////////////////////////////////////////////////////////////

} // namespace NYT::NYson

#define STRING_INL_H_
#include "string-inl.h"
#undef STRING_INL_H_
