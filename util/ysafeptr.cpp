#include "ysafeptr.h"

#ifdef CHECK_YPTR2
Y_POD_THREAD(bool)
IObjectBase::DisableThreadCheck;
#endif
////////////////////////////////////////////////////////////////////////////////////////////////////
void IObjectBase::ReleaseObjComplete(int nMask) {
    if ((ObjData & 0x3fffffff) == 0 && RefData == 0) {
        assert((ObjData & 0x40000000) == 0); // object not being invalidated
        delete this;
    } else if ((ObjData & nMask) == 0) {
        if (ObjData & 0x40000000) {
            // object is already being invalidated
            // possible when no CObj left and object is invalidated and during this all CMObj are also out
            return;
        }
        ObjData |= 0xc0000000;
        AddRef();
        DestroyContents();
        assert((ObjData & nMask) == 0); // otherwise empty constructor is adding CObjs on self
        ObjData &= ~0x40000000;
        ReleaseRef();
    }
}
////////////////////////////////////////////////////////////////////////////////////////////////////
void IObjectBase::ReleaseRefComplete() {
    assert(RefData == 0);
    if ((ObjData & 0x3fffffff) == 0) {
        assert((ObjData & 0x40000000) == 0); // object not being invalidated
        delete this;
    }
}
