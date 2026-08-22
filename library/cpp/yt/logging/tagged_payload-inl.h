#ifndef TAGGED_PAYLOAD_INL_H_
#error "Direct inclusion of this file is not allowed, include tagged_payload.h"
// For the sake of sane code completion.
#include "tagged_payload.h"
#endif

#include <library/cpp/yt/assert/assert.h>

#include <cstring>

namespace NYT::NLogging {

////////////////////////////////////////////////////////////////////////////////

inline TSharedRef TTaggedPayloadBuilder::Flush()
{
    return Buffer_.Slice(0, GetLength());
}

template <class T>
void TTaggedPayloadBuilder::AppendPod(const T& value)
{
    ::memcpy(Preallocate(sizeof(value)), &value, sizeof(value));
    Advance(sizeof(value));
}

template <class T>
void TTaggedPayloadBuilder::WritePodAt(size_t offset, const T& value)
{
    YT_ASSERT(offset + sizeof(value) <= GetLength());
    ::memcpy(Begin_ + offset, &value, sizeof(value));
}

////////////////////////////////////////////////////////////////////////////////

inline TStringBuilderBase* TTaggedPayloadWriter::BeginMessage() &
{
    YT_ASSERT(!MessageStarted_);
    MessageStarted_ = true;
    ReserveLengthPrefix();
    return &Builder_;
}

inline TTaggedPayloadWriter& TTaggedPayloadWriter::EndMessage() &
{
    YT_ASSERT(MessageStarted_ && !MessageEnded_);
    MessageEnded_ = true;
    BackpatchLengthPrefix();
    return *this;
}

inline TStringBuilderBase* TTaggedPayloadWriter::BeginTag(TStringBuf key) &
{
    return DoBeginTag(key, /*wellKnown*/ false);
}

inline TStringBuilderBase* TTaggedPayloadWriter::BeginWellKnownTag(TStringBuf key) &
{
    return DoBeginTag(key, /*wellKnown*/ true);
}

inline TStringBuilderBase* TTaggedPayloadWriter::DoBeginTag(TStringBuf key, bool wellKnown)
{
    YT_ASSERT(MessageEnded_ && !InTag_);
    // High bit reserved for the well-known flag.
    YT_ASSERT(key.size() < WellKnownTagFlag);
    InTag_ = true;
    // Reserve the key-size prefix, the key bytes, and the value-size prefix in a single
    // capacity check; the value is then formatted in place and EndTag backpatches its size.
    auto keySize = static_cast<ui32>(key.size()) | (wellKnown ? WellKnownTagFlag : 0);
    char* ptr = Builder_.Preallocate(2 * sizeof(ui32) + key.size());
    PrefixOffset_ = Builder_.GetLength() + sizeof(ui32) + key.size();
    ::memcpy(ptr, &keySize, sizeof(keySize));
    ::memcpy(ptr + sizeof(keySize), key.data(), key.size());
    // The value-size prefix is left uninitialized; EndTag overwrites it.
    Builder_.Advance(2 * sizeof(ui32) + key.size());
    return &Builder_;
}

inline TTaggedPayloadWriter& TTaggedPayloadWriter::EndTag() &
{
    YT_ASSERT(InTag_);
    InTag_ = false;
    BackpatchLengthPrefix();
    return *this;
}

inline TTaggedPayloadWriter& TTaggedPayloadWriter::AppendTags(TLoggingTagListPayloadView tags) &
{
    YT_ASSERT(MessageEnded_ && !InTag_);
    Builder_.AppendString(tags.Underlying());
    return *this;
}

namespace NDetail {

////////////////////////////////////////////////////////////////////////////////

//! Appends into an external |std::string| rather than owning a buffer. Until #Finish the
//! string may be longer than what has been written, holding uninitialized spare capacity.
class TTaggedPayloadStringBuilder
    : public TStringBuilderBase
{
public:
    explicit TTaggedPayloadStringBuilder(std::string* buffer)
        : Buffer_(buffer)
        , Offset_(buffer->size())
    { }

    void Finish()
    {
        YT_ASSERT(Offset_ + GetLength() <= Buffer_->size());
        Buffer_->resize(Offset_ + GetLength());
    }

private:
    std::string* const Buffer_;
    const size_t Offset_;

    void DoReset() override
    { }

    void DoReserve(size_t newLength) override
    {
        ResizeUninitialized(*Buffer_, Offset_ + newLength);
        auto capacity = Buffer_->capacity();
        ResizeUninitialized(*Buffer_, capacity);
        Begin_ = Buffer_->data() + Offset_;
        End_ = Buffer_->data() + capacity;
    }
};

////////////////////////////////////////////////////////////////////////////////

} // namespace NDetail

template <class TFormatter>
void TTaggedPayloadWriter::AppendTag(
    TLoggingTagListPayload* tags,
    TStringBuf key,
    const TFormatter& formatter)
{
    // High bit reserved for the well-known flag.
    YT_ASSERT(key.size() < WellKnownTagFlag);

    auto& buffer = tags->Underlying();
    auto tagOffset = buffer.size();

    try {
        // Every appended byte is overwritten below, so skip zero-filling them.
        auto keySize = static_cast<ui32>(key.size());
        ResizeUninitialized(buffer, tagOffset + 2 * sizeof(ui32) + key.size());
        char* ptr = buffer.data() + tagOffset;
        ::memcpy(ptr, &keySize, sizeof(keySize));
        ::memcpy(ptr + sizeof(keySize), key.data(), key.size());
        // The value-size prefix is left uninitialized; it is backpatched below.
        auto prefixOffset = tagOffset + sizeof(ui32) + key.size();

        NDetail::TTaggedPayloadStringBuilder builder(&buffer);
        formatter(&builder);
        builder.Finish();

        auto valueLength = buffer.size() - prefixOffset - sizeof(ui32);
        // A truncated prefix would desync the reader for every subsequent tag.
        YT_ASSERT(valueLength <= Max<ui32>());
        auto valueSize = static_cast<ui32>(valueLength);
        // NB: The builder may have reallocated, so recompute the data pointer.
        ::memcpy(buffer.data() + prefixOffset, &valueSize, sizeof(valueSize));
    } catch (...) {
        // Leave #tags exactly as it was; a half-written tag would desync the reader.
        buffer.resize(tagOffset);
        throw;
    }
}

inline TTaggedLogEventPayload TTaggedPayloadWriter::Finish() &
{
    YT_ASSERT(MessageEnded_ && !InTag_);
    return TTaggedLogEventPayload(Builder_.Flush());
}

inline void TTaggedPayloadWriter::ReserveLengthPrefix()
{
    PrefixOffset_ = Builder_.GetLength();
    // Reserve a zeroed ui32; backpatched by BackpatchLengthPrefix.
    Builder_.AppendPod<ui32>(0);
}

inline void TTaggedPayloadWriter::BackpatchLengthPrefix()
{
    Builder_.WritePodAt<ui32>(PrefixOffset_, Builder_.GetLength() - PrefixOffset_ - sizeof(ui32));
}

////////////////////////////////////////////////////////////////////////////////

template <class TBuilder>
void FormatTaggedPayload(TBuilder* builder, const TTaggedLogEventPayload& payload)
{
    TTaggedPayloadReader reader(payload);
    builder->AppendString(reader.ReadMessage());
    // Well-known tags are last by construction (see #TWellKnownTaggedLoggingGuard), hence single-pass.
    bool parenOpen = false;
    while (auto tag = reader.TryReadTag()) {
        if (tag->IsWellKnown) {
            if (parenOpen) {
                builder->AppendChar(')');
                parenOpen = false;
            }
            builder->AppendChar('\n');
            builder->AppendString(tag->Value);
        } else {
            builder->AppendString(parenOpen ? ", "_sb : " ("_sb);
            parenOpen = true;
            builder->AppendString(tag->Key);
            builder->AppendString(": "_sb);
            builder->AppendString(tag->Value);
        }
    }
    if (parenOpen) {
        builder->AppendChar(')');
    }
}

////////////////////////////////////////////////////////////////////////////////

} // namespace NYT::NLogging
