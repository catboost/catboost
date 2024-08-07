#pragma once

#include "typetraits.h"

/**
 * Helper type that can be used as one of the parameters in function declaration
 * to limit the number of types this function can be called with.
 *
 * Example usage:
 * @code
 * void CharOnlyFunction(TExplicitType<char> value);
 * void AnythingFunction(char value);
 *
 * CharOnlyFunction('c'); // Works.
 * CharOnlyFunction(1); // Compilation error.
 * CharOnlyFunction(1ull); // Compilation error.
 *
 * AnythingFunction('c'); // Works.
 * AnythingFunction(1); // Works.
 * AnythingFunction(1ull); // Works.
 * @endcode
 */
template <class T>
class TExplicitType {
public:
    template <class OtherT>
    TExplicitType(const OtherT& value Y_LIFETIME_BOUND, std::enable_if_t<std::is_same<OtherT, T>::value>* = nullptr) noexcept
        : Value_(value)
    {
    }

    const T& Value() const noexcept {
        return Value_;
    }

    operator const T&() const noexcept {
        return Value_;
    }

private:
    const T& Value_;
};
