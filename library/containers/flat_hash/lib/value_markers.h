#pragma once

#include "concepts/value_marker.h"

#include <type_traits>
#include <tuple>

namespace NFlatHash {

namespace NSet {

template <auto Value>
struct TStaticValueMarker {
    using value_type = decltype(Value);

    constexpr auto Create() const noexcept {
        return Value;
    }

    template <class U>
    bool Equals(const U& rhs) const {
        return Value == rhs;
    }
};

static_assert(NConcepts::ValueMarkerV<TStaticValueMarker<5>>);

template <class T>
class TEqValueMarker {
public:
    using value_type = T;

    template <class V, class = std::enable_if_t<std::is_constructible_v<T, std::decay_t<V>>>>
    TEqValueMarker(V&& v) : Value_(std::forward<V>(v)) {}

    TEqValueMarker(const TEqValueMarker&) = default;
    TEqValueMarker(TEqValueMarker&&) = default;

    TEqValueMarker& operator=(const TEqValueMarker&) = default;
    TEqValueMarker& operator=(TEqValueMarker&&) = default;

    const T& Create() const noexcept {
        return Value_;
    }

    template <class U>
    bool Equals(const U& rhs) const {
        return Value_ == rhs;
    }

private:
    T Value_;
};

static_assert(NConcepts::ValueMarkerV<TEqValueMarker<int>>);

}  // namespace NSet

namespace NMap {

template <auto Key, class T>
class TStaticValueMarker {
    static_assert(std::is_default_constructible_v<T>);

public:
    using value_type = std::pair<decltype(Key), T>;

    TStaticValueMarker() = default;

    TStaticValueMarker(const TStaticValueMarker&) {}
    TStaticValueMarker(TStaticValueMarker&&) {}

    TStaticValueMarker& operator=(const TStaticValueMarker&) noexcept { return *this; }
    TStaticValueMarker& operator=(TStaticValueMarker&&) noexcept { return *this; }

    std::pair<decltype(Key), const T&> Create() const noexcept { return { Key, Value_ }; }

    template <class U>
    bool Equals(const U& rhs) const {
        return Key == rhs.first;
    }

private:
    T Value_;
};

static_assert(NConcepts::ValueMarkerV<TStaticValueMarker<5, int>>);

template <class Key, class T>
class TEqValueMarker {
    static_assert(std::is_default_constructible_v<T>);

public:
    using value_type = std::pair<Key, T>;

    template <class V, class = std::enable_if_t<std::is_constructible_v<Key, std::decay_t<V>>>>
    TEqValueMarker(V&& v) : Key_(std::forward<V>(v)) {}

    TEqValueMarker(const TEqValueMarker& vm)
        : Key_(vm.Key_) {}
    TEqValueMarker(TEqValueMarker&& vm) noexcept(std::is_nothrow_move_constructible_v<Key>
                                                 && std::is_nothrow_constructible_v<T>)
        : Key_(std::move(vm.Key_)) {}

    TEqValueMarker& operator=(const TEqValueMarker& vm) {
        Key_ = vm.Key_;
        return *this;
    }
    TEqValueMarker& operator=(TEqValueMarker&& vm) noexcept(std::is_nothrow_move_assignable_v<Key>) {
        Key_ = std::move(vm.Key_);
        return *this;
    }

    auto Create() const noexcept { return std::tie(Key_, Value_); }

    template <class U>
    bool Equals(const U& rhs) const {
        return Key_ == rhs.first;
    }

private:
    Key Key_;
    T Value_;
};

static_assert(NConcepts::ValueMarkerV<TEqValueMarker<int, int>>);

}  // namespace NMap

}  // namespace NFlatHash
