#pragma once

#include <stddef.h>
#include <util/system/compiler.h>
#include <util/system/yassert.h>
#include <util/system/defaults.h>
#include <util/system/tls.h>

////////////////////////////////////////////////////////////////////////////////////////////////////
// There are different templates of pointers:
//   1. Simple pointers.
//   2. TPtr with refereces.
//   3. TObj/TMObj with ownership. After destruction of a TObj the object it referenced to is
//      cleaned up and marked as non valid. Similarly does TMobj organizing the parallel ownership
//      of an object.
//
// Limitations:
//   1. It may be necessary to use BASIC_REGISTER_CLASS() in .cpp files to be able to use a
//      pointer to a forward declared class.
//   2. It's prohibited to override the 'new' operator, since the standard 'delete' will be used
//      for destruction of objects (because of 'delete this').
////////////////////////////////////////////////////////////////////////////////////////////////////

#if defined(_MSC_VER) && defined(_DEBUG)
    #include <util/system/winint.h>
    #define CHECK_YPTR2
#endif

struct IBinSaver;

class IObjectBase {
private:
#ifdef CHECK_YPTR2
    static Y_POD_THREAD(bool) DisableThreadCheck;
    void CheckThreadId() {
        if (dwThreadId == 0)
            dwThreadId = GetCurrentThreadId();
        else
            Y_ASSERT(dwThreadId == GetCurrentThreadId() || DisableThreadCheck);
    }
    void AddRef() {
        CheckThreadId();
        ++RefData;
    }
    void AddObj(int nRef) {
        CheckThreadId();
        ObjData += nRef;
    }
#else
    void CheckThreadId() {
    }
    void AddRef() {
        ++RefData;
    }
    void AddObj(int nRef) {
        ObjData += nRef;
    }
#endif
    void ReleaseRefComplete();
    void ReleaseObjComplete(int nMask);
    void DecRef() {
        CheckThreadId();
        --RefData;
    }
    void DecObj(int nRef) {
        CheckThreadId();
        ObjData -= nRef;
    }
    void ReleaseRef() {
        CheckThreadId();
        --RefData;
        if (RefData == 0)
            ReleaseRefComplete();
    }
    void ReleaseObj(int nRef, int nMask) {
        CheckThreadId();
        ObjData -= nRef;
        if ((ObjData & nMask) == 0)
            ReleaseObjComplete(nMask);
    }

protected:
#ifdef CHECK_YPTR2
    DWORD dwThreadId;
#endif
    ui32 ObjData;
    ui32 RefData;
    // function should clear contents of object, easy to implement via consequent calls to
    // destructor and constructor, this function should not be called directly, use Clear()
    virtual void DestroyContents() = 0;
    virtual ~IObjectBase() = default;
    inline void CopyValidFlag(const IObjectBase& a) {
        ObjData &= 0x7fffffff;
        ObjData |= a.ObjData & 0x80000000;
    }

public:
    IObjectBase()
        : ObjData(0)
        , RefData(0)
    {
#ifdef CHECK_YPTR2
        dwThreadId = 0;
#endif
    }
    // do not copy refcount when copy object
    IObjectBase(const IObjectBase& a)
        : ObjData(0)
        , RefData(0)
    {
#ifdef CHECK_YPTR2
        dwThreadId = 0;
#endif
        CopyValidFlag(a);
    }
    IObjectBase& operator=(const IObjectBase& a) {
        CopyValidFlag(a);
        return *this;
    }
#ifdef CHECK_YPTR2
    static void SetThreadCheckMode(bool val) {
        DisableThreadCheck = !val;
    }
    void ResetThreadId() {
        Y_ASSERT(RefData == 0 && ObjData == 0); // can reset thread check only for ref free objects
        dwThreadId = 0;
    }
#else
    static void SetThreadCheckMode(bool) {
    }
    void ResetThreadId() {
    }
#endif

    // class name of derived class
    virtual const char* GetClassName() const = 0;

    ui32 IsRefInvalid() const {
        return (ObjData & 0x80000000);
    }
    ui32 IsRefValid() const {
        return !IsRefInvalid();
    }
    // reset data in class to default values, saves RefCount from destruction
    void Clear() {
        AddRef();
        DestroyContents();
        DecRef();
    }

    virtual int operator&(IBinSaver&) {
        return 0;
    }

    struct TRefO {
        void AddRef(IObjectBase* pObj) {
            pObj->AddObj(1);
        }
        void DecRef(IObjectBase* pObj) {
            pObj->DecObj(1);
        }
        void Release(IObjectBase* pObj) {
            pObj->ReleaseObj(1, 0x000fffff);
        }
    };
    struct TRefM {
        void AddRef(IObjectBase* pObj) {
            pObj->AddObj(0x100000);
        }
        void DecRef(IObjectBase* pObj) {
            pObj->DecObj(0x100000);
        }
        void Release(IObjectBase* pObj) {
            pObj->ReleaseObj(0x100000, 0x3ff00000);
        }
    };
    struct TRef {
        void AddRef(IObjectBase* pObj) {
            pObj->AddRef();
        }
        void DecRef(IObjectBase* pObj) {
            pObj->DecRef();
        }
        void Release(IObjectBase* pObj) {
            pObj->ReleaseRef();
        }
    };
    friend struct IObjectBase::TRef;
    friend struct IObjectBase::TRefO;
    friend struct IObjectBase::TRefM;
};
////////////////////////////////////////////////////////////////////////////////////////////////////
// macro that helps to create neccessary members for proper operation of refcount system
// if class needs special destructor, use CFundament
#define OBJECT_METHODS(classname)                                 \
public:                                                           \
    virtual const char* GetClassName() const override {           \
        return #classname;                                        \
    }                                                             \
    static IObjectBase* NewSaveLoadNullItem() {                   \
        return new classname();                                   \
    }                                                             \
                                                                  \
protected:                                                        \
    virtual void DestroyContents() override {                     \
        this->~classname();                                       \
        int nHoldRefs = this->RefData, nHoldObjs = this->ObjData; \
        new (this) classname();                                   \
        this->RefData += nHoldRefs;                               \
        this->ObjData += nHoldObjs;                               \
    }                                                             \
                                                                  \
private:                                                          \
    Y_SEMICOLON_GUARD

#define OBJECT_NOCOPY_METHODS(classname) OBJECT_METHODS(classname)
#define BASIC_REGISTER_CLASS(classname)                                              \
    Y_PRAGMA_DIAGNOSTIC_PUSH                                                         \
    Y_PRAGMA_NO_UNUSED_FUNCTION                                                      \
    template <>                                                                      \
    IObjectBase* CastToObjectBaseImpl<classname>(classname * p, void*) {             \
        return p;                                                                    \
    }                                                                                \
    template <>                                                                      \
    classname* CastToUserObjectImpl<classname>(IObjectBase * p, classname*, void*) { \
        return dynamic_cast<classname*>(p);                                          \
    }                                                                                \
    Y_PRAGMA_DIAGNOSTIC_POP

////////////////////////////////////////////////////////////////////////////////////////////////////
template <class TUserObj>
IObjectBase* CastToObjectBaseImpl(TUserObj* p, void*);
template <class TUserObj>
IObjectBase* CastToObjectBaseImpl(TUserObj* p, IObjectBase*) {
    return p;
}
template <class TUserObj>
TUserObj* CastToUserObjectImpl(IObjectBase* p, TUserObj*, void*);
template <class TUserObj>
TUserObj* CastToUserObjectImpl(IObjectBase* _p, TUserObj*, IObjectBase*) {
    return dynamic_cast<TUserObj*>(_p);
}
template <class TUserObj>
inline IObjectBase* CastToObjectBase(TUserObj* p) {
    return CastToObjectBaseImpl(p, p);
}
template <class TUserObj>
inline const IObjectBase* CastToObjectBase(const TUserObj* p) {
    return p;
}
template <class TUserObj>
inline TUserObj* CastToUserObject(IObjectBase* p, TUserObj* pu) {
    return CastToUserObjectImpl(p, pu, pu);
}
////////////////////////////////////////////////////////////////////////////////////////////////////
// TObject - base object for reference counting, TUserObj - user object name
// TRef - struct with AddRef/DecRef/Release methods for refcounting to use
template <class TUserObj, class TRef>
class TPtrBase {
private:
    TUserObj* ptr;

    void AddRef(TUserObj* _ptr) {
        TRef p;
        if (_ptr)
            p.AddRef(CastToObjectBase(_ptr));
    }
    void DecRef(TUserObj* _ptr) {
        TRef p;
        if (_ptr)
            p.DecRef(CastToObjectBase(_ptr));
    }
    void Release(TUserObj* _ptr) {
        TRef p;
        if (_ptr)
            p.Release(CastToObjectBase(_ptr));
    }

protected:
    void SetObject(TUserObj* _ptr) {
        TUserObj* pOld = ptr;
        ptr = _ptr;
        AddRef(ptr);
        Release(pOld);
    }

public:
    TPtrBase()
        : ptr(nullptr)
    {
    }
    TPtrBase(TUserObj* _ptr)
        : ptr(_ptr)
    {
        AddRef(ptr);
    }
    TPtrBase(const TPtrBase& a)
        : ptr(a.ptr)
    {
        AddRef(ptr);
    }
    ~TPtrBase() {
        Release(ptr);
    }

    void Set(TUserObj* _ptr) {
        SetObject(_ptr);
    }
    TUserObj* Extract() {
        TUserObj* pRes = ptr;
        DecRef(ptr);
        ptr = nullptr;
        return pRes;
    }

    const char* GetClassName() const {
        return ptr->GetClassName();
    }

    // assignment operators
    TPtrBase& operator=(TUserObj* _ptr) {
        Set(_ptr);
        return *this;
    }
    TPtrBase& operator=(const TPtrBase& a) {
        Set(a.ptr);
        return *this;
    }
    // access
    TUserObj* operator->() const {
        return ptr;
    }
    operator TUserObj*() const {
        return ptr;
    }
    TUserObj* Get() const {
        return ptr;
    }
    IObjectBase* GetBarePtr() const {
        return CastToObjectBase(ptr);
    }
    int operator&(IBinSaver& f);
};
////////////////////////////////////////////////////////////////////////////////////////////////////
template <class T>
inline bool IsValid(T* p) {
    return p != nullptr && !CastToObjectBase(p)->IsRefInvalid();
}
template <class T, class TRef>
inline bool IsValid(const TPtrBase<T, TRef>& p) {
    return p.Get() && !p.GetBarePtr()->IsRefInvalid();
}
////////////////////////////////////////////////////////////////////////////////////////////////////
#define BASIC_PTR_DECLARE(TPtrName, TRef)        \
    template <class T>                           \
    class TPtrName: public TPtrBase<T, TRef> {   \
        using CBase = TPtrBase<T, TRef>;         \
                                                 \
    public:                                      \
        using CDestType = T;                     \
        TPtrName() {                             \
        }                                        \
        TPtrName(T* _ptr)                        \
            : CBase(_ptr)                        \
        {                                        \
        }                                        \
        TPtrName(const TPtrName& a)              \
            : CBase(a)                           \
        {                                        \
        }                                        \
        TPtrName& operator=(T* _ptr) {           \
            this->Set(_ptr);                     \
            return *this;                        \
        }                                        \
        TPtrName& operator=(const TPtrName& a) { \
            this->SetObject(a.Get());            \
            return *this;                        \
        }                                        \
        int operator&(IBinSaver& f) {            \
            return (*(CBase*)this) & (f);        \
        }                                        \
    };

BASIC_PTR_DECLARE(TPtr, IObjectBase::TRef)
BASIC_PTR_DECLARE(TObj, IObjectBase::TRefO)
BASIC_PTR_DECLARE(TMObj, IObjectBase::TRefM)
// misuse guard
template <class T>
inline bool IsValid(TObj<T>* p) {
    return p->YouHaveMadeMistake();
}
template <class T>
inline bool IsValid(TPtr<T>* p) {
    return p->YouHaveMadeMistake();
}
template <class T>
inline bool IsValid(TMObj<T>* p) {
    return p->YouHaveMadeMistake();
}
////////////////////////////////////////////////////////////////////////////////////////////////////
// assumes base class is IObjectBase
template <class T>
class TDynamicCast {
    T* ptr;

public:
    template <class TT>
    TDynamicCast(TT* _ptr) {
        ptr = dynamic_cast<T*>(CastToObjectBase(_ptr));
    }
    template <class TT>
    TDynamicCast(const TT* _ptr) {
        ptr = dynamic_cast<T*>(CastToObjectBase(const_cast<TT*>(_ptr)));
    }
    template <class T1, class T2>
    TDynamicCast(const TPtrBase<T1, T2>& _ptr) {
        ptr = dynamic_cast<T*>(_ptr.GetBarePtr());
    }
    operator T*() const {
        return ptr;
    }
    T* operator->() const {
        return ptr;
    }
    T* Get() const {
        return ptr;
    }
};
template <class T>
inline bool IsValid(const TDynamicCast<T>& p) {
    return IsValid(p.Get());
}
