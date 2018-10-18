#pragma once

#include <memory>

namespace NPrivate {
    enum class ERefOrObjectHolderType {
        Reference,
        Object,
        SharedPtr
    };

    template <typename TRefOrObject, ERefOrObjectHolderType Type>
    struct TRefOrObjectHolderBase {
    };

    template <typename TRefOrObject>
    struct TRefOrObjectHolderBase<TRefOrObject, ERefOrObjectHolderType::Reference> {
        using TObject = typename std::remove_reference<TRefOrObject>::type;
        using TObjectStorage = TObject*;

        TRefOrObjectHolderBase(TObject& object)
            : Storage(&object)
        {
        }

        TObject* Get() {
            return Storage;
        }

        TObject* Get() const {
            return Storage;
        }

        TObjectStorage Storage;
    };

    template <typename TRefOrObject>
    struct TRefOrObjectHolderBase<TRefOrObject, ERefOrObjectHolderType::Object> {
        using TObject = typename std::remove_reference<TRefOrObject>::type;
        using TObjectStorage = TObject;

        TRefOrObjectHolderBase(TObject& object)
            : Storage(std::move(object))
        {
        }

        TObject* Get() {
            return &Storage;
        }

        TObject* Get() const {
            return &Storage;
        }

        TObjectStorage Storage;
    };

    template <typename TRefOrObject>
    struct TRefOrObjectHolderBase<TRefOrObject, ERefOrObjectHolderType::SharedPtr> {
        using TObject = typename std::remove_reference<TRefOrObject>::type;
        using TObjectStorage = std::shared_ptr<TObject>;

        TRefOrObjectHolderBase(TObject& object)
            : Storage(std::make_shared<TObject>(std::move(object)))
        {
        }

        TObject* Get() {
            return Storage.get();
        }

        TObject* Get() const {
            return Storage.get();
        }

        TObjectStorage Storage;
    };

    // store object in shared_ptr if we have rvalue reference or just save a pointer
    template <typename TRefOrObject, ERefOrObjectHolderType NotRefHolderType,
              typename TBase_ = TRefOrObjectHolderBase<TRefOrObject, std::is_reference<TRefOrObject>::value ? ERefOrObjectHolderType::Reference : NotRefHolderType>>
    struct TRefOrSmthHolder : TBase_ {
        using TBase = TBase_;
        using TObject = typename TBase::TObject;
        using TObjectStorage = typename TBase::TObjectStorage;

        TRefOrSmthHolder(TObject& object)
            : TBase(object)
        {
        }

        TObject& operator*() {
            return *this->Get();
        }

        TObject& operator*() const {
            return *this->Get();
        }

        TObject* operator->() {
            return this->Get();
        }

        TObject* operator->() const {
            return this->Get();
        }
    };

}

//! Store object as field if we have rvalue reference or just save a pointer
template <typename TRefOrObject,
          typename TBase = ::NPrivate::TRefOrSmthHolder<TRefOrObject, ::NPrivate::ERefOrObjectHolderType::Object>>
struct TRefOrObjectHolder : TBase {
    TRefOrObjectHolder(typename TBase::TObject& object)
        : ::NPrivate::TRefOrSmthHolder<TRefOrObject, ::NPrivate::ERefOrObjectHolderType::Object>(object)
    {
    }
};

// store object in shared_ptr if we have rvalue reference or just save a pointer
template <typename TRefOrObject,
          typename TBase = ::NPrivate::TRefOrSmthHolder<TRefOrObject, ::NPrivate::ERefOrObjectHolderType::SharedPtr>>
struct TRefOrObjectSharedHolder : TBase {
    TRefOrObjectSharedHolder(typename TBase::TObject& object)
        : ::NPrivate::TRefOrSmthHolder<TRefOrObject, ::NPrivate::ERefOrObjectHolderType::SharedPtr>(object)
    {
    }
};
