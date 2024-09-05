#pragma once

#include "store_policy.h"
#include "typetraits.h"

namespace NPrivate {
    template <class Range>
    class TReverseRangeStorage {
    public:
        TReverseRangeStorage(Range&& range)
            : Base_(std::forward<Range>(range))
        {
        }

        decltype(auto) Base() const {
            return *Base_.Ptr();
        }

        decltype(auto) Base() {
            return *Base_.Ptr();
        }

    private:
        TAutoEmbedOrPtrPolicy<Range> Base_;
    };

    template <class Range>
    constexpr bool HasReverseIterators(i32, decltype(std::declval<Range>().rbegin())*) {
        return true;
    }

    template <class Range>
    constexpr bool HasReverseIterators(char, std::nullptr_t*) {
        return false;
    }

    template <class Range, bool hasReverseIterators = HasReverseIterators<Range>((i32)0, nullptr)>
    class TReverseRangeBase: public TReverseRangeStorage<Range> {
        using TBase = TReverseRangeStorage<Range>;

    public:
        using TBase::Base;
        using TBase::TBase;

        auto begin() const {
            return Base().rbegin();
        }

        auto end() const {
            return Base().rend();
        }

        auto begin() {
            return Base().rbegin();
        }

        auto end() {
            return Base().rend();
        }
    };

    template <class Range>
    class TReverseRangeBase<Range, false>: public TReverseRangeStorage<Range> {
        using TBase = TReverseRangeStorage<Range>;

    public:
        using TBase::Base;
        using TBase::TBase;

        auto begin() const {
            using std::end;
            return std::make_reverse_iterator(end(Base()));
        }

        auto end() const {
            using std::begin;
            return std::make_reverse_iterator(begin(Base()));
        }

        auto begin() {
            using std::end;
            return std::make_reverse_iterator(end(Base()));
        }

        auto end() {
            using std::begin;
            return std::make_reverse_iterator(begin(Base()));
        }
    };

    template <class Range>
    class TReverseRange: public TReverseRangeBase<Range> {
        using TBase = TReverseRangeBase<Range>;

    public:
        using TBase::Base;
        using TBase::TBase;

        TReverseRange(TReverseRange&&) = default;
        TReverseRange(const TReverseRange&) = default;

        auto rbegin() const {
            using std::begin;
            return begin(Base());
        }

        auto rend() const {
            using std::end;
            return end(Base());
        }

        auto rbegin() {
            using std::begin;
            return begin(Base());
        }

        auto rend() {
            using std::end;
            return end(Base());
        }
    };
} // namespace NPrivate

/**
 * Provides a reverse view into the provided container.
 *
 * Example usage:
 * @code
 * for(auto&& value: Reversed(container)) {
 *     // use value here.
 * }
 * @endcode
 *
 * @param cont                          Container to provide a view into. Must be an lvalue.
 * @returns                             A reverse view into the provided container.
 */
template <class Range>
constexpr ::NPrivate::TReverseRange<Range> Reversed(Range&& range) {
    return ::NPrivate::TReverseRange<Range>(std::forward<Range>(range));
}
