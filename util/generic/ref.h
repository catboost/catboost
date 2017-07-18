#pragma once

#include "ptr.h"
#include "typetraits.h"

/* a potentially unsafe class (but no more so than TStringBuf) which can
 * be initialized either by references (without ownership) or by smart
 * or allocated pointers (with ownership and reference counting) */

template <typename TPSmartPtr, typename T = typename TPSmartPtr::TValueType>
struct TSmartPtrTraits {
    enum {
        IsSmartPtr = std::is_base_of<TPointerBase<TPSmartPtr, T>, TPSmartPtr>::value,
    };
};

template <typename TPSmartPtr>
class TSmartRef: public TPointerBase<TSmartRef<TPSmartPtr>, std::enable_if_t<TSmartPtrTraits<TPSmartPtr>::IsSmartPtr, typename TPSmartPtr::TValueType>> {
public:
    using TDVal = typename TPSmartPtr::TValueType;
    using TDSmartPtr = TPSmartPtr;
    using TDSelf = TSmartRef;

private:
    TDSmartPtr SmartPtr_;
    TDVal* Ptr_;

protected:
    // don't use when non-trivial copy semantics apply to TDSmartPtr
    TSmartRef(TDVal* ptr, const TDSmartPtr& sptr)
        : SmartPtr_(sptr)
        , Ptr_(ptr)
    {
    }

public:
    TSmartRef(TDVal& ref)
        : Ptr_(&ref)
    {
    }

    TSmartRef(TDVal* ptr = 0)
        : SmartPtr_(ptr)
        , Ptr_(ptr)
    {
    }

    TSmartRef(const TPSmartPtr& ptr)
        : SmartPtr_(ptr)
        , Ptr_(SmartPtr_.Get())
    {
    }

    // copy constructor
    TSmartRef(const TDSelf& obj)
        : SmartPtr_(obj.SmartPtr_)
        , Ptr_(SmartPtr_.Get() ? SmartPtr_.Get() : obj.Ptr_)
    {
    }

    // make sure TPOther is a smart pointer and its value type derives from TDVal
    template <typename TPOther, typename TPOtherVal = typename TPOther::TValueType>
    struct TOtherTraits {
        enum {
            IsCompatType = std::is_base_of<TDVal, TPOtherVal>::value,
            IsCompatSmartPtr = IsCompatType && TSmartPtrTraits<TPOther, TPOtherVal>::IsSmartPtr,
        };
    };

    template <typename TPOther>
    TSmartRef(const TPOther& ptr, std::enable_if_t<TOtherTraits<TPOther>::IsCompatSmartPtr, bool> = false)
        : SmartPtr_(ptr)
        , Ptr_(SmartPtr_.Get())
    {
    }

    template <typename TPOther>
    friend class TSmartRef;

    template <typename TPOther>
    TSmartRef(const TSmartRef<TPOther>& obj, std::enable_if_t<TOtherTraits<TPOther>::IsCompatType, bool> = false)
        : SmartPtr_(obj.SmartPtr_)
        , Ptr_(SmartPtr_.Get() ? SmartPtr_.Get() : obj.Ptr_)
    {
    }

    // assignment operator
    TDSelf& operator=(TDSelf ref) {
        Swap(ref);
        return *this;
    }

    // TPointerCommon
    TDVal* Get() const {
        return Ptr_;
    }

    const TDSmartPtr& SmartPtr() const {
        return SmartPtr_;
    }

    // other
    void Reset() {
        Reset(TDSelf());
    }

    void Reset(const TDSelf& ref) {
        *this = ref;
    }

    void Swap(TDSelf& obj) {
        SmartPtr_.Swap(obj.SmartPtr_);
        DoSwap(Ptr_, obj.Ptr_);
    }
};

// template "typedef"
template <typename T, typename C = TSimpleCounter, typename D = TDelete>
struct TSharedRef: public TSmartRef<TSharedPtr<T, C, D>> {
    using TDSmartPtr = TSharedPtr<T, C, D>;
    using TDSmartRef = TSmartRef<TDSmartPtr>;

    TSharedRef(T& ref)
        : TDSmartRef(ref)
    {
    }

    TSharedRef(T* ptr = nullptr)
        : TDSmartRef(ptr)
    {
    }

    TSharedRef(const TDSmartPtr& ptr)
        : TDSmartRef(ptr)
    {
    }

    TSharedRef(const TDSmartRef& obj)
        : TDSmartRef(obj.Get(), obj.SmartPtr())
    {
    }

    TSharedRef(const TAutoPtr<T, D>& ptr)
        : TDSmartRef(ptr)
    {
    }

    // make sure U derives from T

    template <typename U>
    TSharedRef(const TSharedPtr<U, C, D>& ptr, std::enable_if_t<std::is_base_of<T, U>::value, bool> = false)
        : TDSmartRef(ptr)
    {
    }

    template <typename U>
    TSharedRef(const TSmartRef<TSharedPtr<U, C, D>>& obj, std::enable_if_t<std::is_base_of<T, U>::value, bool> = false)
        : TDSmartRef(obj.Get(), obj.SmartPtr())
    {
    }
};
