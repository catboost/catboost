#pragma once

#include <util/generic/store_policy.h>

#include <util/stream/zerocopy_output.h>

#include <util/system/compiler.h>

#include <string>

namespace NYT {

////////////////////////////////////////////////////////////////////////////////

//! Drop-in, |std::string|-backed replacements for util's |TStringOutput| /
//! |TStringStream| (util/stream/str.h).
//!
//! The util adapters are hard-wired to |TString| (they take a |TString&| / own
//! a |TString|), which forces any string they touch to stay |TString|. These
//! mirror the same |IOutputStream| / zero-copy interfaces over |std::string|,
//! so a buffer that previously had to be |TString| solely to feed one of those
//! streams can become |std::string|.
//!
//! Only the output/owning variants live here: for *reading* from a
//! |std::string| / |TStringBuf|, util's |TMemoryInput| already works (it views
//! raw bytes and involves no |TString|).

//! An output stream that appends to a referenced |std::string|.
class TStdStringOutput
    : public IZeroCopyOutput
{
public:
    //! Constructs an output stream appending to \p string.
    /*!
     *  As with |TStringOutput|, the stream keeps a reference to \p string, so
     *  the caller must keep it alive while the stream is in use.
     */
    explicit TStdStringOutput(std::string& string Y_LIFETIME_BOUND) noexcept;

    TStdStringOutput(TStdStringOutput&&) noexcept = default;

    ~TStdStringOutput() override;

    //! Reserves \p size additional characters in the output string.
    void Reserve(size_t size);

    void Swap(TStdStringOutput& other) noexcept;

protected:
    size_t DoNext(void** ptr) override;
    void DoUndo(size_t len) override;
    void DoWrite(const void* buf, size_t len) override;
    void DoWriteC(char c) override;

private:
    std::string* String_;
};

////////////////////////////////////////////////////////////////////////////////

//! An output stream that owns a |std::string| (the writing half of util's
//! |TStringStream|). Read the accumulated bytes back via |Str()|, or wrap it in
//! a |TMemoryInput|.
class TStdStringStream
    : private TEmbedPolicy<std::string>
    , public TStdStringOutput
{
    using TEmbeddedString = TEmbedPolicy<std::string>;

public:
    TStdStringStream();
    explicit TStdStringStream(const std::string& string);
    explicit TStdStringStream(std::string&& string);
    TStdStringStream(const TStdStringStream& other);
    TStdStringStream(TStdStringStream&& other) noexcept;

    TStdStringStream& operator=(const TStdStringStream& other);
    TStdStringStream& operator=(TStdStringStream&& other) noexcept;

    ~TStdStringStream() override;

    //! Returns whether this stream contains any data.
    explicit operator bool() const noexcept;

    //! Returns the string this stream is writing into.
    std::string& Str() & noexcept;
    const std::string& Str() const& noexcept;
    std::string&& Str() && noexcept;

    //! Returns a pointer to the (null-terminated) character data.
    const char* Data() const noexcept;

    //! Returns the total number of characters in the stream.
    size_t Size() const noexcept;

    //! Returns whether the underlying string is empty.
    Y_PURE_FUNCTION bool Empty() const noexcept;

    using TStdStringOutput::Reserve;

    //! Clears the underlying string.
    void Clear();
};

////////////////////////////////////////////////////////////////////////////////

} // namespace NYT
