#include "stream.h"

#include <util/generic/bitops.h>
#include <util/generic/string.h>
#include <util/generic/utility.h>
#include <util/system/yassert.h>

namespace NYT {

////////////////////////////////////////////////////////////////////////////////

static constexpr size_t MinBufferGrowSize = 16;

////////////////////////////////////////////////////////////////////////////////

TStdStringOutput::TStdStringOutput(std::string& string) noexcept
    : String_(&string)
{ }

TStdStringOutput::~TStdStringOutput() = default;

void TStdStringOutput::Reserve(size_t size)
{
    String_->reserve(String_->size() + size);
}

void TStdStringOutput::Swap(TStdStringOutput& other) noexcept
{
    DoSwap(String_, other.String_);
}

size_t TStdStringOutput::DoNext(void** ptr)
{
    if (String_->size() == String_->capacity()) {
        String_->reserve(FastClp2(String_->capacity() + MinBufferGrowSize));
    }
    size_t previousSize = String_->size();
    ResizeUninitialized(*String_, String_->capacity());
    *ptr = String_->data() + previousSize;
    return String_->size() - previousSize;
}

void TStdStringOutput::DoUndo(size_t len)
{
    Y_ABORT_UNLESS(len <= String_->size(), "trying to undo more bytes than actually written");
    String_->resize(String_->size() - len);
}

void TStdStringOutput::DoWrite(const void* buf, size_t len)
{
    String_->append(static_cast<const char*>(buf), len);
}

void TStdStringOutput::DoWriteC(char c)
{
    String_->push_back(c);
}

////////////////////////////////////////////////////////////////////////////////

TStdStringStream::TStdStringStream()
    : TEmbeddedString()
    , TStdStringOutput(*TEmbeddedString::Ptr())
{ }

TStdStringStream::TStdStringStream(const std::string& string)
    : TEmbeddedString(string)
    , TStdStringOutput(*TEmbeddedString::Ptr())
{ }

TStdStringStream::TStdStringStream(std::string&& string)
    : TEmbeddedString(std::move(string))
    , TStdStringOutput(*TEmbeddedString::Ptr())
{ }

TStdStringStream::TStdStringStream(const TStdStringStream& other)
    : TEmbeddedString(other.Str())
    , TStdStringOutput(*TEmbeddedString::Ptr())
{ }

TStdStringStream::TStdStringStream(TStdStringStream&& other) noexcept
    : TEmbeddedString(std::move(other).Str())
    , TStdStringOutput(*TEmbeddedString::Ptr())
{ }

TStdStringStream& TStdStringStream::operator=(const TStdStringStream& other)
{
    // The embedded string reference remains valid; only its contents change.
    Str() = other.Str();
    return *this;
}

TStdStringStream& TStdStringStream::operator=(TStdStringStream&& other) noexcept
{
    // The embedded string reference remains valid; only its contents change.
    Str() = std::move(other).Str();
    return *this;
}

TStdStringStream::~TStdStringStream() = default;

TStdStringStream::operator bool() const noexcept
{
    return !Empty();
}

std::string& TStdStringStream::Str() & noexcept
{
    return *Ptr();
}

const std::string& TStdStringStream::Str() const& noexcept
{
    return *Ptr();
}

std::string&& TStdStringStream::Str() && noexcept
{
    return std::move(*Ptr());
}

const char* TStdStringStream::Data() const noexcept
{
    return Ptr()->data();
}

size_t TStdStringStream::Size() const noexcept
{
    return Ptr()->size();
}

bool TStdStringStream::Empty() const noexcept
{
    return Ptr()->empty();
}

void TStdStringStream::Clear()
{
    Str().clear();
}

////////////////////////////////////////////////////////////////////////////////

} // namespace NYT
