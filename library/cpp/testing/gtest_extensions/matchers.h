#pragma once

#include <util/generic/string.h>

#include <gtest/gtest.h>
#include <gmock/gmock.h>

namespace testing {
    /**
     * When matching `const TStringBuf&`, implicitly convert other strings and string views to `Eq` matchers.
     */
    template <typename T, typename TT>
    class Matcher<const TBasicStringBuf<T, TT>&>: public internal::MatcherBase<const TBasicStringBuf<T, TT>&> {
    public:
        Matcher() {
        }

        explicit Matcher(const MatcherInterface<const TBasicStringBuf<T, TT>&>* impl)
            : internal::MatcherBase<const TBasicStringBuf<T, TT>&>(impl) {
        }

        template <typename M, typename = typename std::remove_reference<M>::type::is_gtest_matcher>
        Matcher(M&& m)
            : internal::MatcherBase<const TBasicStringBuf<T, TT>&>(std::forward<M>(m)) {
        }

        Matcher(const TBasicString<T, TT>& s) {
            *this = Eq(TBasicStringBuf<T, TT>(s));
        }

        Matcher(const T* s) {
            *this = Eq(TBasicStringBuf<T, TT>(s));
        }

        Matcher(TBasicStringBuf<T, TT> s) {
            *this = Eq(s);
        }
    };

    /**
     * When matching `TBasicBuf`, implicitly convert other strings and string views to `Eq` matchers.
     */
    template <typename T, typename TT>
    class Matcher<TBasicStringBuf<T, TT>>: public internal::MatcherBase<TBasicStringBuf<T, TT>> {
    public:
        Matcher() {
        }

        explicit Matcher(const MatcherInterface <TBasicStringBuf<T, TT>>* impl)
            : internal::MatcherBase<TBasicStringBuf<T, TT>>(impl) {
        }

        explicit Matcher(const MatcherInterface<const TBasicStringBuf<T, TT>&>* impl)
            : internal::MatcherBase<TBasicStringBuf<T, TT>>(impl) {
        }

        template <typename M, typename = typename std::remove_reference<M>::type::is_gtest_matcher>
        Matcher(M&& m)
            : internal::MatcherBase<TBasicStringBuf<T, TT>>(std::forward<M>(m)) {
        }

        Matcher(const TBasicString<T, TT>& s) {
            *this = Eq(TBasicString<T, TT>(s));
        }

        Matcher(const T* s) {
            *this = Eq(TBasicString<T, TT>(s));
        }

        Matcher(TBasicStringBuf<T, TT> s) {
            *this = Eq(s);
        }
    };

    /**
     * When matching `const TString&`, implicitly convert other strings and string views to `Eq` matchers.
     */
    template <typename T, typename TT>
    class Matcher<const TBasicString<T, TT>&>: public internal::MatcherBase<const TBasicString<T, TT>&> {
    public:
        Matcher() {
        }

        explicit Matcher(const MatcherInterface<const TBasicString<T, TT>&>* impl)
            : internal::MatcherBase<const TBasicString<T, TT>&>(impl) {
        }

        Matcher(const TBasicString<T, TT>& s) {
            *this = Eq(s);
        }

        template <typename M, typename = typename std::remove_reference<M>::type::is_gtest_matcher>
        Matcher(M&& m)
            : internal::MatcherBase<const TBasicString<T, TT>&>(std::forward<M>(m)) {
        }

        Matcher(const T* s) {
            *this = Eq(TBasicString<T, TT>(s));
        }
    };

    /**
     * When matching `TString`, implicitly convert other strings and string views to `Eq` matchers.
     */
    template <typename T, typename TT>
    class Matcher<TBasicString<T, TT>>: public internal::MatcherBase<TBasicString<T, TT>> {
    public:
        Matcher() {
        }

        explicit Matcher(const MatcherInterface <TBasicString<T, TT>>* impl)
            : internal::MatcherBase<TBasicString<T, TT>>(impl) {
        }

        explicit Matcher(const MatcherInterface<const TBasicString<T, TT>&>* impl)
            : internal::MatcherBase<TBasicString<T, TT>>(impl) {
        }

        template <typename M, typename = typename std::remove_reference<M>::type::is_gtest_matcher>
        Matcher(M&& m)
            : internal::MatcherBase<TBasicString<T, TT>>(std::forward<M>(m)) {
        }

        Matcher(const TBasicString<T, TT>& s) {
            *this = Eq(s);
        }

        Matcher(const T* s) {
            *this = Eq(TBasicString<T, TT>(s));
        }
    };
}
