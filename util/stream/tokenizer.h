#pragma once

#include "input.h"

#include <util/generic/buffer.h>
#include <util/generic/mem_copy.h>
#include <util/generic/strbuf.h>
#include <util/system/compiler.h>
#include <util/system/yassert.h>

/**
 * @addtogroup Streams
 * @{
 */

/**
 * Simple stream tokenizer. Splits the stream into tokens that are available
 * via iterator interface.
 *
 * @tparam TEndOfToken                  Predicate for token delimiter characters.
 * @see TEol
 */
template <typename TEndOfToken>
class TStreamTokenizer {
public:
    class TIterator {
    public:
        inline TIterator(TStreamTokenizer* const parent)
            : Parent_(parent)
            , AtEnd_(!Parent_->Next(Data_, Len_))
        {
        }

        inline TIterator() noexcept
            : Parent_(nullptr)
            , Data_(nullptr)
            , Len_(0)
            , AtEnd_(true)
        {
        }

        inline ~TIterator() = default;

        inline void operator++() {
            Next();
        }

        inline bool operator==(const TIterator& l) const noexcept {
            return AtEnd_ == l.AtEnd_;
        }

        inline bool operator!=(const TIterator& l) const noexcept {
            return !(*this == l);
        }

        /**
         * @return          Return null-terminated character array with current token.
         *                  The pointer may be invalid after iterator increment.
         */
        inline const char* Data() const noexcept {
            Y_ASSERT(!AtEnd_);

            return Data_;
        }

        /**
         * @return          Length of current token.
         */
        inline size_t Length() const noexcept {
            Y_ASSERT(!AtEnd_);

            return Len_;
        }

        inline TIterator* operator->() noexcept {
            return this;
        }

        inline TStringBuf operator*() noexcept {
            return TStringBuf{Data_, Len_};
        }

    private:
        inline void Next() {
            Y_ASSERT(Parent_);

            AtEnd_ = !Parent_->Next(Data_, Len_);
        }

    private:
        TStreamTokenizer* const Parent_;
        char* Data_;
        size_t Len_;
        bool AtEnd_;
    };

    inline TStreamTokenizer(IInputStream* const input, const TEndOfToken& eot = TEndOfToken(),
                            const size_t initial = 1024)
        : Input_(input)
        , Buf_(initial)
        , Cur_(BufBegin())
        , End_(BufBegin())
        , Eot_(eot)
    {
        CheckBuf();
    }

    inline bool Next(char*& buf, size_t& len) {
        char* it = Cur_;

        while (true) {
            do {
                while (it != End_) {
                    if (Eot_(*it)) {
                        *it = '\0';

                        buf = Cur_;
                        len = it - Cur_;
                        Cur_ = it + 1;

                        return true;
                    } else {
                        ++it;
                    }
                }

                if (Fill() == 0 && End_ != BufEnd()) {
                    *it = '\0';

                    buf = Cur_;
                    len = it - Cur_;
                    Cur_ = End_;

                    return len;
                }
            } while (it != BufEnd());

            Y_ASSERT(it == BufEnd());
            Y_ASSERT(End_ == BufEnd());

            const size_t blen = End_ - Cur_;
            if (Cur_ == BufBegin()) {
                Y_ASSERT(blen == Buf_.Capacity());

                /*
                 * do reallocate
                 */

                Buf_.Reserve(Buf_.Capacity() * 4);
                CheckBuf();
            } else {
                /*
                 * do move
                 */

                MemMove(BufBegin(), Cur_, blen);
            }

            Cur_ = BufBegin();
            End_ = Cur_ + blen;
            it = End_;
        }
    }

    inline TIterator begin() {
        return TIterator{this};
    }

    inline TIterator end() noexcept {
        return {};
    }

private:
    inline size_t Fill() {
        const size_t avail = BufEnd() - End_;
        const size_t bytesRead = Input_->Read(End_, avail);

        End_ += bytesRead;

        return bytesRead;
    }

    inline char* BufBegin() noexcept {
        return Buf_.Data();
    }

    inline char* BufEnd() noexcept {
        return Buf_.Data() + Buf_.Capacity();
    }

    inline void CheckBuf() const {
        if (!Buf_.Data()) {
            throw std::bad_alloc();
        }
    }

private:
    IInputStream* const Input_;
    TBuffer Buf_;
    char* Cur_;
    char* End_;
    TEndOfToken Eot_;
};

/**
 * Predicate for `TStreamTokenizer` that uses '\\n' as a delimiter.
 */
struct TEol {
    inline bool operator()(char ch) const noexcept {
        return ch == '\n';
    }
};

/** @} */
