#pragma once

#include <memory>
#include <type_traits>
#include <initializer_list>

namespace NMaybe {
    struct TInPlace {};

    template <class T, bool = std::is_trivially_destructible<T>::value>
    struct TStorageBase {
        constexpr TStorageBase() noexcept
            : NullState_('\0')
        {
        }

        template <class... Args>
        constexpr TStorageBase(TInPlace, Args&&... args)
            : Data_(std::forward<Args>(args)...)
            , Defined_(true)
        {
        }

        constexpr TStorageBase(TStorageBase&&) = default;
        constexpr TStorageBase(const TStorageBase&) = default;

        ~TStorageBase() = default;

        TStorageBase& operator=(const TStorageBase&) = default;
        TStorageBase& operator=(TStorageBase&&) = default;

        union {
            char NullState_;
            T Data_;
        };
        bool Defined_ = false;
    };

    template <class T>
    struct TStorageBase<T, false> {
        constexpr TStorageBase() noexcept
            : NullState_('\0')
        {
        }

        template <class... Args>
        constexpr TStorageBase(TInPlace, Args&&... args)
            : Data_(std::forward<Args>(args)...)
            , Defined_(true)
        {
        }

        constexpr TStorageBase(TStorageBase&&) = default;
        constexpr TStorageBase(const TStorageBase&) = default;

        ~TStorageBase() {
            if (this->Defined_) {
                this->Data_.~T();
            }
        }

        TStorageBase& operator=(const TStorageBase&) = default;
        TStorageBase& operator=(TStorageBase&&) = default;

        union {
            char NullState_;
            T Data_;
        };
        bool Defined_ = false;
    };

    // -------------------- COPY CONSTRUCT --------------------

    template <class T, bool = std::is_trivially_copy_constructible<T>::value>
    struct TCopyBase: TStorageBase<T> {
        using TStorageBase<T>::TStorageBase;
    };

    template <class T>
    struct TCopyBase<T, false>: TStorageBase<T> {
        using TStorageBase<T>::TStorageBase;

        constexpr TCopyBase() = default;
        constexpr TCopyBase(const TCopyBase& rhs) {
            if (rhs.Defined_) {
                new (std::addressof(this->Data_)) T(rhs.Data_);
                this->Defined_ = true;
            }
        }
        constexpr TCopyBase(TCopyBase&&) = default;
        TCopyBase& operator=(const TCopyBase&) = default;
        TCopyBase& operator=(TCopyBase&&) = default;
    };

    // -------------------- MOVE CONSTRUCT --------------------

    template <class T, bool = std::is_trivially_move_constructible<T>::value>
    struct TMoveBase: TCopyBase<T> {
        using TCopyBase<T>::TCopyBase;
    };

    template <class T>
    struct TMoveBase<T, false>: TCopyBase<T> {
        using TCopyBase<T>::TCopyBase;

        constexpr TMoveBase() noexcept = default;
        constexpr TMoveBase(const TMoveBase&) = default;
        constexpr TMoveBase(TMoveBase&& rhs) noexcept(std::is_nothrow_move_constructible<T>::value) {
            if (rhs.Defined_) {
                new (std::addressof(this->Data_)) T(std::move(rhs.Data_));
                this->Defined_ = true;
            }
        }
        TMoveBase& operator=(const TMoveBase&) = default;
        TMoveBase& operator=(TMoveBase&&) = default;
    };

    // -------------------- COPY ASSIGN --------------------

    template <class T, bool = std::is_trivially_copy_assignable<T>::value>
    struct TCopyAssignBase: TMoveBase<T> {
        using TMoveBase<T>::TMoveBase;
    };

    template <class T>
    struct TCopyAssignBase<T, false>: TMoveBase<T> {
        using TMoveBase<T>::TMoveBase;

        constexpr TCopyAssignBase() noexcept = default;
        constexpr TCopyAssignBase(const TCopyAssignBase&) = default;
        constexpr TCopyAssignBase(TCopyAssignBase&&) = default;
        TCopyAssignBase& operator=(const TCopyAssignBase& rhs) {
            if (this->Defined_) {
                if (rhs.Defined_) {
                    this->Data_ = rhs.Data_;
                } else {
                    this->Data_.~T();
                    this->Defined_ = false;
                }
            } else if (rhs.Defined_) {
                new (std::addressof(this->Data_)) T(rhs.Data_);
                this->Defined_ = true;
            }
            return *this;
        }
        TCopyAssignBase& operator=(TCopyAssignBase&&) = default;
    };

    // -------------------- MOVE ASSIGN --------------------

    template <class T, bool = std::is_trivially_move_assignable<T>::value>
    struct TMoveAssignBase: TCopyAssignBase<T> {
        using TCopyAssignBase<T>::TCopyAssignBase;
    };

    template <class T>
    struct TMoveAssignBase<T, false>: TCopyAssignBase<T> {
        using TCopyAssignBase<T>::TCopyAssignBase;

        constexpr TMoveAssignBase() noexcept = default;
        constexpr TMoveAssignBase(const TMoveAssignBase&) = default;
        constexpr TMoveAssignBase(TMoveAssignBase&&) = default;
        TMoveAssignBase& operator=(const TMoveAssignBase&) = default;
        TMoveAssignBase& operator=(TMoveAssignBase&& rhs) noexcept(
            std::is_nothrow_move_assignable<T>::value&&
                std::is_nothrow_move_constructible<T>::value)
        {
            if (this->Defined_) {
                if (rhs.Defined_) {
                    this->Data_ = std::move(rhs.Data_);
                } else {
                    this->Data_.~T();
                    this->Defined_ = false;
                }
            } else if (rhs.Defined_) {
                new (std::addressof(this->Data_)) T(std::move(rhs.Data_));
                this->Defined_ = true;
            }
            return *this;
        }
    };
}

template <class T>
using TMaybeBase = NMaybe::TMoveAssignBase<T>;
