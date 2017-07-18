#pragma once

/**
 * @class TNonCopyable
 *
 * Inherit your class from `TNonCopyable` if you want to make it noncopyable.
 *
 * It's better than DECLARE_NOCOPY macro because:
 *     1. You don't have to type your class name twice.
 *     2. You are able to use your class without implicitly defined ctors.
 *
 * Example usage:
 * @code
 * class Foo: private TNonCopyable {
 *     // ...
 * };
 * @endcode
 */

namespace NNonCopyable { // protection from unintended ADL
    class TNonCopyable {
    protected:
        TNonCopyable(const TNonCopyable&) = delete;
        TNonCopyable& operator=(const TNonCopyable&) = delete;

        TNonCopyable() = default;
        ~TNonCopyable() = default;
    };

    class TMoveOnly {
    protected:
        TMoveOnly(TMoveOnly&&) noexcept = default;
        TMoveOnly& operator=(TMoveOnly&&) noexcept = default;

        TMoveOnly(const TMoveOnly&) = delete;
        TMoveOnly& operator=(const TMoveOnly&) = delete;

        TMoveOnly() = default;
        ~TMoveOnly() = default;
    };
}

using TNonCopyable = NNonCopyable::TNonCopyable;
using TMoveOnly = NNonCopyable::TMoveOnly;
