#pragma once

#include <util/generic/fwd.h>
#include <util/generic/vector.h>
#include <util/generic/map.h>

#include <cstddef>
#include <type_traits>

/*

A file with declarations of enumeration-related functions.
It doesn't contains definitions. To generate them you have to add

    GENERATE_ENUM_SERIALIZATION_WITH_HEADER(your_header_with_your_enum.h)
or
    GENERATE_ENUM_SERIALIZATION(your_header_with_your_enum.h)

in your ya.make

@see https://st.yandex-team.ru/IGNIETFERRO-333
@see https://wiki.yandex-team.ru/PoiskovajaPlatforma/Build/WritingCmakefiles/#generate-enum-with-header

*/

/**
 * Returns number of distinct items in enum or enum class
 *
 * @tparam EnumT     enum type
 */
template <typename EnumT>
Y_CONST_FUNCTION constexpr size_t GetEnumItemsCount();

namespace NEnumSerializationRuntime {
    namespace NDetail {
        template <typename EEnum>
        struct TSelectEnumRepresentationType;

        template <typename TEnumType, typename TRepresentationType, class TStorage = TVector<TRepresentationType>>
        class TMappedArrayView;

        template <typename TEnumType, typename TRepresentationType, typename TValueType, class TStorage = TMap<TRepresentationType, TValueType>>
        class TMappedDictView;
    }

    /// Class with behaviour similar to TMap<EnumT, TValueType>
    template <typename EnumT, typename TValueType>
    using TMappedDictView = NDetail::TMappedDictView<EnumT, typename NDetail::TSelectEnumRepresentationType<EnumT>::TType, TValueType>;

    /// Class with behaviour similar to TVector<EnumT>
    template <typename EnumT>
    using TMappedArrayView = NDetail::TMappedArrayView<EnumT, typename NDetail::TSelectEnumRepresentationType<EnumT>::TType>;

    /**
     * Returns names for items in enum or enum class
     *
     * @tparam EnumT     enum type
     */
    template <typename EnumT>
    TMappedDictView<EnumT, TString> GetEnumNamesImpl();
    /**
     * Returns unique items in enum or enum class
     *
     * @tparam EnumT     enum type
     */
    template <typename EnumT>
    ::NEnumSerializationRuntime::TMappedArrayView<EnumT> GetEnumAllValuesImpl();

    /**
     * Returns human-readable comma-separated list of names in enum or enum class
     *
     * @tparam EnumT     enum type
     */
    template <typename EnumT>
    const TString& GetEnumAllNamesImpl();

    /**
     * Returns C++ identifiers for items in enum or enum class
     *
     * @tparam EnumT     enum type
     */
    template <typename EnumT>
    const TVector<TString>& GetEnumAllCppNamesImpl();

    /**
     * Converts @c e to a string. Works like @c ToString(e) function, but returns @c TStringBuf instead of @c TString.
     * Thus works slightly faster and usually avoids any dynamic memory allocation.
     * @throw yexception is case of unknown enum value
     */
    template <typename EnumT>
    TStringBuf ToStringBuf(EnumT e);
}

/**
 * Returns names for items in enum or enum class
 *
 * @tparam EnumT     enum type
 */
template <typename EnumT>
Y_CONST_FUNCTION ::NEnumSerializationRuntime::TMappedDictView<EnumT, TString> GetEnumNames() {
    return ::NEnumSerializationRuntime::GetEnumNamesImpl<EnumT>();
}

/**
 * Returns unique items in enum or enum class
 *
 * @tparam EnumT     enum type
 */
template <typename EnumT>
Y_CONST_FUNCTION ::NEnumSerializationRuntime::TMappedArrayView<EnumT> GetEnumAllValues() {
    return ::NEnumSerializationRuntime::GetEnumAllValuesImpl<EnumT>();
}

/**
 * Returns human-readable comma-separated list of names in enum or enum class
 *
 * @tparam EnumT     enum type
 */
template <typename EnumT>
Y_CONST_FUNCTION const TString& GetEnumAllNames() {
    return ::NEnumSerializationRuntime::GetEnumAllNamesImpl<EnumT>();
}

/**
 * Returns C++ identifiers for items in enum or enum class
 *
 * @tparam EnumT     enum type
 */
template <typename EnumT>
Y_CONST_FUNCTION const TVector<TString>& GetEnumAllCppNames() {
    return ::NEnumSerializationRuntime::GetEnumAllCppNamesImpl<EnumT>();
}

namespace NEnumSerializationRuntime {
    namespace NDetail {
        /// Checks that the `From` type can be promoted up to the `To` type without losses
        template <typename From, typename To>
        struct TIsPromotable: public std::is_same<std::common_type_t<From, To>, To> {
            static_assert(std::is_integral<From>::value, "`From` type has to be an integer");
            static_assert(std::is_integral<To>::value, "`To` type has to be an integer");
        };

        /// Selects enum representation type. Works like std::underlying_type_t<>, but promotes small types up to `int`
        template <typename EEnum>
        struct TSelectEnumRepresentationType {
            using TUnderlyingType = std::underlying_type_t<EEnum>;
            using TIsSigned = std::is_signed<TUnderlyingType>;
            using TRepresentationType = std::conditional_t<
                TIsSigned::value,
                std::conditional_t<
                    TIsPromotable<TUnderlyingType, int>::value,
                    int,
                    long long>,
                std::conditional_t<
                    TIsPromotable<TUnderlyingType, unsigned>::value,
                    unsigned,
                    unsigned long long>>;
            using TType = TRepresentationType;
            static_assert(sizeof(TUnderlyingType) <= sizeof(TType), "size of `TType` is not smaller than the size of `TUnderlyingType`");
        };

        template <typename TEnumType, typename TRepresentationType>
        class TMappedViewBase {
            static_assert(sizeof(std::underlying_type_t<TEnumType>) <= sizeof(TRepresentationType), "Internal type is probably too small to represent all possible values");

        public:
            static constexpr TEnumType CastFromRepresentationType(const TRepresentationType key) noexcept {
                return static_cast<TEnumType>(key);
            }

            static constexpr TRepresentationType CastToRepresentationType(const TEnumType key) noexcept {
                return static_cast<TRepresentationType>(key);
            }
        };

        /// Wrapper class with behaviour similar to TVector<EnumT>
        ///
        /// @tparam TEnumType            enum type at the external interface
        /// @tparam TRepresentationType  designated underlying type of enum
        /// @tparam TStorage             internal container type
        template <typename TEnumType, typename TRepresentationType, class TStorage>
        class TMappedArrayView: public TMappedViewBase<TEnumType, TRepresentationType> {
        public:
            using value_type = TEnumType;

        public:
            TMappedArrayView(const TStorage& a) noexcept
                : Ref(a)
            {
            }

            class TIterator {
            public:
                using TSlaveIteratorType = typename TStorage::const_iterator;

                using difference_type = std::ptrdiff_t;
                using value_type = TEnumType;
                using pointer = const TEnumType*;
                using reference = const TEnumType&;
                using iterator_category = std::bidirectional_iterator_tag;

            public:
                TIterator(TSlaveIteratorType it)
                    : Slave(std::move(it))
                {
                }

                bool operator==(const TIterator& it) const {
                    return Slave == it.Slave;
                }

                bool operator!=(const TIterator& it) const {
                    return !(*this == it);
                }

                TEnumType operator*() const {
                    return TMappedArrayView::CastFromRepresentationType(*Slave);
                }

                TIterator& operator++() {
                    ++Slave;
                    return *this;
                }

                TIterator& operator--() {
                    --Slave;
                    return *this;
                }

                TIterator operator++(int) {
                    auto temp = Slave;
                    ++Slave;
                    return temp;
                }

                TIterator operator--(int) {
                    auto temp = Slave;
                    --Slave;
                    return temp;
                }

            private:
                TSlaveIteratorType Slave;
            };

            TIterator begin() const {
                return Ref.begin();
            }

            TIterator end() const {
                return Ref.end();
            }

            size_t size() const {
                return Ref.size();
            }

            Y_PURE_FUNCTION bool empty() const {
                return Ref.empty();
            }

            TEnumType at(size_t index) const {
                return this->CastFromRepresentationType(Ref.at(index));
            }

            TEnumType operator[](size_t index) const {
                return this->CastFromRepresentationType(Ref[index]);
            }

            // Allocate container and copy view's content into it
            template <template <class...> class TContainer = TVector>
            TContainer<TEnumType> Materialize() const {
                return {begin(), end()};
            }

        private:
            const TStorage& Ref;
        };

        /// Wrapper class with behaviour similar to TMap<EnumT, TValueType>
        ///
        /// @tparam TEnumType            enum type at the external interface
        /// @tparam TRepresentationType  designated underlying type of enum
        /// @tparam TValueType           mapped value
        /// @tparam TStorage             internal container type
        template <typename TEnumType, typename TRepresentationType, typename TValueType, class TStorage>
        class TMappedDictView: public TMappedViewBase<TEnumType, TRepresentationType> {
        public:
            using TMappedItemType = std::pair<const TEnumType, const TValueType&>;

            class TDereferenceResultHolder {
            public:
                TDereferenceResultHolder(const TRepresentationType enumValue, const TValueType& payload) noexcept
                    : Data(TMappedDictView::CastFromRepresentationType(enumValue), payload)
                {
                }

                const TMappedItemType* operator->() const noexcept {
                    return &Data;
                }

            private:
                TMappedItemType Data;
            };

            TMappedDictView(const TStorage& m) noexcept
                : Ref(m)
            {
            }

            class TIterator {
            public:
                using TSlaveIteratorType = typename TStorage::const_iterator;

                using difference_type = std::ptrdiff_t;
                using value_type = TMappedItemType;
                using pointer = const TMappedItemType*;
                using reference = const TMappedItemType&;
                using iterator_category = std::bidirectional_iterator_tag;

            public:
                TIterator(TSlaveIteratorType it)
                    : Slave(std::move(it))
                {
                }

                bool operator==(const TIterator& it) const {
                    return Slave == it.Slave;
                }

                bool operator!=(const TIterator& it) const {
                    return !(*this == it);
                }

                TDereferenceResultHolder operator->() const {
                    return {Slave->first, Slave->second};
                }

                TMappedItemType operator*() const {
                    return {TMappedDictView::CastFromRepresentationType(Slave->first), Slave->second};
                }

                TIterator& operator++() {
                    ++Slave;
                    return *this;
                }

                TIterator& operator--() {
                    --Slave;
                    return *this;
                }

                TIterator operator++(int) {
                    auto temp = Slave;
                    ++Slave;
                    return temp;
                }

                TIterator operator--(int) {
                    auto temp = Slave;
                    --Slave;
                    return temp;
                }

            private:
                TSlaveIteratorType Slave;
            };

            TIterator begin() const {
                return Ref.begin();
            }

            TIterator end() const {
                return Ref.end();
            }

            size_t size() const {
                return Ref.size();
            }

            Y_PURE_FUNCTION bool empty() const {
                return Ref.empty();
            }

            bool contains(const TEnumType key) const {
                return Ref.contains(this->CastToRepresentationType(key));
            }

            TIterator find(const TEnumType key) const {
                return Ref.find(this->CastToRepresentationType(key));
            }

            const TValueType& at(const TEnumType key) const {
                return Ref.at(this->CastToRepresentationType(key));
            }

            // Allocate container and copy view's content into it
            template <template <class...> class TContainer = TMap>
            TContainer<TEnumType, TValueType> Materialize() const {
                return {begin(), end()};
            }

        private:
            const TStorage& Ref;
        };
    }
}
