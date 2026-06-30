#pragma once

/**
 * @class TNonCopyable
 *
 * Inherit your class from `TNonCopyable` if you want to make it noncopyable.
 *
 * Example usage:
 * @code
 * class Foo: private TNonCopyable {
 *     // ...
 * };
 * @endcode
 */

namespace NNonCopyable { // protection from unintended ADL
    struct TNonCopyable {
        TNonCopyable(const TNonCopyable&) = delete;
        TNonCopyable& operator=(const TNonCopyable&) = delete;

        TNonCopyable() = default;
        ~TNonCopyable() = default;
    };

    struct TMoveOnly {
        TMoveOnly(TMoveOnly&&) noexcept = default;
        TMoveOnly& operator=(TMoveOnly&&) noexcept = default;

        TMoveOnly(const TMoveOnly&) = delete;
        TMoveOnly& operator=(const TMoveOnly&) = delete;

        TMoveOnly() = default;
        ~TMoveOnly() = default;
    };
} // namespace NNonCopyable

using TNonCopyable = NNonCopyable::TNonCopyable;
using TMoveOnly = NNonCopyable::TMoveOnly;
