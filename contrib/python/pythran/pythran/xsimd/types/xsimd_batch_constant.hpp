/***************************************************************************
 * Copyright (c) Johan Mabille, Sylvain Corlay, Wolf Vollprecht and         *
 * Martin Renou                                                             *
 * Copyright (c) QuantStack                                                 *
 * Copyright (c) Serge Guelton                                              *
 *                                                                          *
 * Distributed under the terms of the BSD 3-Clause License.                 *
 *                                                                          *
 * The full license is in the file LICENSE, distributed with this software. *
 ****************************************************************************/

#ifndef XSIMD_BATCH_CONSTANT_HPP
#define XSIMD_BATCH_CONSTANT_HPP

#include "./xsimd_batch.hpp"
#include "./xsimd_utils.hpp"

namespace xsimd
{
    /**
     * @brief batch of boolean constant
     *
     * Abstract representation of a batch of boolean constants.
     *
     * @tparam batch_type the type of the associated batch values.
     * @tparam Values boolean constant represented by this batch
     **/
    template <class batch_type, bool... Values>
    struct batch_bool_constant
    {

    public:
        static constexpr std::size_t size = sizeof...(Values);
        using arch_type = typename batch_type::arch_type;
        using value_type = bool;
        static_assert(sizeof...(Values) == batch_type::size, "consistent batch size");

        constexpr operator batch_bool<typename batch_type::value_type, arch_type>() const noexcept { return { Values... }; }

        constexpr bool get(size_t i) const noexcept
        {
            return std::array<value_type, size> { { Values... } }[i];
        }

        static constexpr int mask() noexcept
        {
            return mask_helper(0, static_cast<int>(Values)...);
        }

    private:
        static constexpr int mask_helper(int acc) noexcept { return acc; }

        template <class... Tys>
        static constexpr int mask_helper(int acc, int mask, Tys... masks) noexcept
        {
            return mask_helper(acc | mask, (masks << 1)...);
        }

        struct logical_or
        {
            constexpr bool operator()(bool x, bool y) const { return x || y; }
        };
        struct logical_and
        {
            constexpr bool operator()(bool x, bool y) const { return x && y; }
        };
        struct logical_xor
        {
            constexpr bool operator()(bool x, bool y) const { return x ^ y; }
        };

        template <class F, class SelfPack, class OtherPack, size_t... Indices>
        static constexpr batch_bool_constant<batch_type, F()(std::tuple_element<Indices, SelfPack>::type::value, std::tuple_element<Indices, OtherPack>::type::value)...>
        apply(detail::index_sequence<Indices...>)
        {
            return {};
        }

        template <class F, bool... OtherValues>
        static constexpr auto apply(batch_bool_constant<batch_type, Values...>, batch_bool_constant<batch_type, OtherValues...>)
            -> decltype(apply<F, std::tuple<std::integral_constant<bool, Values>...>, std::tuple<std::integral_constant<bool, OtherValues>...>>(detail::make_index_sequence<sizeof...(Values)>()))
        {
            static_assert(sizeof...(Values) == sizeof...(OtherValues), "compatible constant batches");
            return apply<F, std::tuple<std::integral_constant<bool, Values>...>, std::tuple<std::integral_constant<bool, OtherValues>...>>(detail::make_index_sequence<sizeof...(Values)>());
        }

    public:
#define MAKE_BINARY_OP(OP, NAME)                                                            \
    template <bool... OtherValues>                                                          \
    constexpr auto operator OP(batch_bool_constant<batch_type, OtherValues...> other) const \
        -> decltype(apply<NAME>(*this, other))                                              \
    {                                                                                       \
        return apply<NAME>(*this, other);                                                   \
    }

        MAKE_BINARY_OP(|, logical_or)
        MAKE_BINARY_OP(||, logical_or)
        MAKE_BINARY_OP(&, logical_and)
        MAKE_BINARY_OP(&&, logical_and)
        MAKE_BINARY_OP(^, logical_xor)

#undef MAKE_BINARY_OP

        constexpr batch_bool_constant<batch_type, !Values...> operator!() const
        {
            return {};
        }

        constexpr batch_bool_constant<batch_type, !Values...> operator~() const
        {
            return {};
        }
    };

    /**
     * @brief batch of integral constants
     *
     * Abstract representation of a batch of integral constants.
     *
     * @tparam batch_type the type of the associated batch values.
     * @tparam Values constants represented by this batch
     **/
    template <class batch_type, typename batch_type::value_type... Values>
    struct batch_constant
    {
        static constexpr std::size_t size = sizeof...(Values);
        using arch_type = typename batch_type::arch_type;
        using value_type = typename batch_type::value_type;
        static_assert(sizeof...(Values) == batch_type::size, "consistent batch size");

        /**
         * @brief Generate a batch of @p batch_type from this @p batch_constant
         */
        inline operator batch_type() const noexcept { return { Values... }; }

        /**
         * @brief Get the @p i th element of this @p batch_constant
         */
        constexpr value_type get(size_t i) const noexcept
        {
            return get(i, std::array<value_type, size> { Values... });
        }

    private:
        constexpr value_type get(size_t i, std::array<value_type, size> const& values) const noexcept
        {
            return values[i];
        }

        struct arithmetic_add
        {
            constexpr value_type operator()(value_type x, value_type y) const { return x + y; }
        };
        struct arithmetic_sub
        {
            constexpr value_type operator()(value_type x, value_type y) const { return x - y; }
        };
        struct arithmetic_mul
        {
            constexpr value_type operator()(value_type x, value_type y) const { return x * y; }
        };
        struct arithmetic_div
        {
            constexpr value_type operator()(value_type x, value_type y) const { return x / y; }
        };
        struct arithmetic_mod
        {
            constexpr value_type operator()(value_type x, value_type y) const { return x % y; }
        };
        struct binary_and
        {
            constexpr value_type operator()(value_type x, value_type y) const { return x & y; }
        };
        struct binary_or
        {
            constexpr value_type operator()(value_type x, value_type y) const { return x | y; }
        };
        struct binary_xor
        {
            constexpr value_type operator()(value_type x, value_type y) const { return x ^ y; }
        };

        template <class F, class SelfPack, class OtherPack, size_t... Indices>
        static constexpr batch_constant<batch_type, F()(std::tuple_element<Indices, SelfPack>::type::value, std::tuple_element<Indices, OtherPack>::type::value)...>
        apply(detail::index_sequence<Indices...>)
        {
            return {};
        }

        template <class F, value_type... OtherValues>
        static constexpr auto apply(batch_constant<batch_type, Values...>, batch_constant<batch_type, OtherValues...>)
            -> decltype(apply<F, std::tuple<std::integral_constant<value_type, Values>...>, std::tuple<std::integral_constant<value_type, OtherValues>...>>(detail::make_index_sequence<sizeof...(Values)>()))
        {
            static_assert(sizeof...(Values) == sizeof...(OtherValues), "compatible constant batches");
            return apply<F, std::tuple<std::integral_constant<value_type, Values>...>, std::tuple<std::integral_constant<value_type, OtherValues>...>>(detail::make_index_sequence<sizeof...(Values)>());
        }

    public:
#define MAKE_BINARY_OP(OP, NAME)                                                       \
    template <value_type... OtherValues>                                               \
    constexpr auto operator OP(batch_constant<batch_type, OtherValues...> other) const \
        -> decltype(apply<NAME>(*this, other))                                         \
    {                                                                                  \
        return apply<NAME>(*this, other);                                              \
    }

        MAKE_BINARY_OP(+, arithmetic_add)
        MAKE_BINARY_OP(-, arithmetic_sub)
        MAKE_BINARY_OP(*, arithmetic_mul)
        MAKE_BINARY_OP(/, arithmetic_div)
        MAKE_BINARY_OP(%, arithmetic_mod)
        MAKE_BINARY_OP(&, binary_and)
        MAKE_BINARY_OP(|, binary_or)
        MAKE_BINARY_OP(^, binary_xor)

#undef MAKE_BINARY_OP

        constexpr batch_constant<batch_type, (value_type)-Values...> operator-() const
        {
            return {};
        }

        constexpr batch_constant<batch_type, (value_type) + Values...> operator+() const
        {
            return {};
        }

        constexpr batch_constant<batch_type, (value_type)~Values...> operator~() const
        {
            return {};
        }
    };

    namespace detail
    {
        template <class batch_type, class G, std::size_t... Is>
        inline constexpr auto make_batch_constant(detail::index_sequence<Is...>) noexcept
            -> batch_constant<batch_type, (typename batch_type::value_type)G::get(Is, sizeof...(Is))...>
        {
            return {};
        }
        template <class batch_type, class G, std::size_t... Is>
        inline constexpr auto make_batch_bool_constant(detail::index_sequence<Is...>) noexcept
            -> batch_bool_constant<batch_type, G::get(Is, sizeof...(Is))...>
        {
            return {};
        }

    } // namespace detail

    /**
     * @brief Build a @c batch_constant out of a generator function
     *
     * @tparam batch_type type of the (non-constant) batch to build
     * @tparam G type used to generate that batch. That type must have a static
     * member @c get that's used to generate the batch constant. Conversely, the
     * generated batch_constant has value `{G::get(0, batch_size), ... , G::get(batch_size - 1, batch_size)}`
     *
     * The following generator produces a batch of `(n - 1, 0, 1, ... n-2)`
     *
     * @code
     * struct Rot
     * {
     *     static constexpr unsigned get(unsigned i, unsigned n)
     *     {
     *         return (i + n - 1) % n;
     *     }
     * };
     * @endcode
     */
    template <class batch_type, class G>
    inline constexpr auto make_batch_constant() noexcept -> decltype(detail::make_batch_constant<batch_type, G>(detail::make_index_sequence<batch_type::size>()))
    {
        return detail::make_batch_constant<batch_type, G>(detail::make_index_sequence<batch_type::size>());
    }

    template <class batch_type, class G>
    inline constexpr auto make_batch_bool_constant() noexcept
        -> decltype(detail::make_batch_bool_constant<batch_type, G>(
            detail::make_index_sequence<batch_type::size>()))
    {
        return detail::make_batch_bool_constant<batch_type, G>(
            detail::make_index_sequence<batch_type::size>());
    }

} // namespace xsimd

#endif
