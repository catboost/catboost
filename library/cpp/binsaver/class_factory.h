#pragma once

#include <typeinfo>
#include <util/generic/hash.h>
#include <util/generic/vector.h>
#include <util/ysafeptr.h>


////////////////////////////////////////////////////////////////////////////////////////////////////
// factory is using RTTI
// objects should inherit T and T must have at least 1 virtual function
template <class T>
class TClassFactory {
public:
    typedef const std::type_info* VFT;

private:
    typedef T* (*newFunc)();
    typedef THashMap<int, newFunc> CTypeNewHash;           // typeID->newFunc()
    typedef THashMap<VFT, int> CTypeIndexHash; // vftable->typeID

    CTypeIndexHash typeIndex;
    CTypeNewHash typeInfo;

    void RegisterTypeBase(int nTypeID, newFunc func, VFT vft);
    static VFT GetObjectType(T* pObject) {
        return &typeid(*pObject);
    }
    int VFT2TypeID(VFT t) {
        CTypeIndexHash::iterator i = typeIndex.find(t);
        if (i != typeIndex.end())
            return i->second;
        for (i = typeIndex.begin(); i != typeIndex.end(); ++i) {
            if (*i->first == *t) {
                typeIndex[t] = i->second;
                return i->second;
            }
        }
        return -1;
    }

public:
    template <class TT>
    void RegisterType(int nTypeID, newFunc func, TT*) {
        RegisterTypeBase(nTypeID, func, &typeid(TT));
    }
    void RegisterTypeSafe(int nTypeID, newFunc func) {
        TPtr<T> pObj = func();
        VFT vft = GetObjectType(pObj);
        RegisterTypeBase(nTypeID, func, vft);
    }
    T* CreateObject(int nTypeID) {
        newFunc f = typeInfo[nTypeID];
        if (f)
            return f();
        return nullptr;
    }
    int GetObjectTypeID(T* pObject) {
        return VFT2TypeID(GetObjectType(pObject));
    }
    template <class TT>
    int GetTypeID(TT* p = 0) {
        (void)p;
        return VFT2TypeID(&typeid(TT));
    }

    void GetAllTypeIDs(TVector<int>& typeIds) const {
        typeIds.clear();
        for (typename CTypeNewHash::const_iterator iter = typeInfo.begin();
             iter != typeInfo.end();
             ++iter) {
            typeIds.push_back(iter->first);
        }
    }
};
////////////////////////////////////////////////////////////////////////////////////////////////////
template <class T>
void TClassFactory<T>::RegisterTypeBase(int nTypeID, newFunc func, VFT vft) {
    if (typeInfo.find(nTypeID) != typeInfo.end()) {
        TObj<IObjectBase> o1 = typeInfo[nTypeID]();
        TObj<IObjectBase> o2 = func();

        // stupid clang warning
        auto& o1v = *o1;
        auto& o2v = *o2;

        if (typeid(o1v) != typeid(o2v)) {
            fprintf(stderr, "IBinSaver: Type ID 0x%08X has been already used\n", nTypeID);
            abort();
        }
    }

    CTypeIndexHash::iterator typeIndexIt = typeIndex.find(vft);
    if (typeIndexIt != typeIndex.end() && nTypeID != typeIndexIt->second) {
        fprintf(stderr, "IBinSaver: class (Type ID 0x%08X) has been already registered (Type ID 0x%08X)\n", nTypeID, typeIndexIt->second);
        abort();
    }
    typeIndex[vft] = nTypeID;
    typeInfo[nTypeID] = func;
}
////////////////////////////////////////////////////////////////////////////////////////////////////
// macro for registering CFundament derivatives
#define REGISTER_CLASS(factory, N, name) factory.RegisterType(N, name::New##name, (name*)0);
#define REGISTER_TEMPL_CLASS(factory, N, name, className) factory.RegisterType(N, name::New##className, (name*)0);
#define REGISTER_CLASS_NM(factory, N, name, nmspace) factory.RegisterType(N, nmspace::name::New##name, (nmspace::name*)0);
