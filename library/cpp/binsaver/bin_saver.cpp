#include "bin_saver.h"

TClassFactory<IObjectBase>* pSaverClasses;
void StartRegisterSaveload() {
    if (!pSaverClasses)
        pSaverClasses = new TClassFactory<IObjectBase>;
}
struct SBasicChunkInit {
    ~SBasicChunkInit() {
        if (pSaverClasses)
            delete pSaverClasses;
    }
} initSaver;

//////////////////////////////////////////////////////////////////////////
void IBinSaver::StoreObject(IObjectBase* pObject) {
    if (pObject) {
        Y_ASSERT(pSaverClasses->GetObjectTypeID(pObject) != -1 && "trying to save unregistered object");
    }

    ui64 ptrId = reinterpret_cast<ui64>(pObject) / sizeof(char);
    if (StableOutput) {
        ui32 id = 0;
        if (pObject) {
            if (!PtrIds.Get())
                PtrIds.Reset(new PtrIdHash);
            PtrIdHash::iterator pFound = PtrIds->find(pObject);
            if (pFound != PtrIds->end())
                id = pFound->second;
            else {
                id = PtrIds->ysize() + 1;
                PtrIds->insert(std::make_pair(pObject, id));
            }
        }
        ptrId = id;
    }

    DataChunk(&ptrId, sizeof(ptrId));
    if (!Objects.Get())
        Objects.Reset(new CObjectsHash);
    if (ptrId != 0 && Objects->find(ptrId) == Objects->end()) {
        ObjectQueue.push_back(pObject);
        (*Objects)[ptrId];
        int typeId = pSaverClasses->GetObjectTypeID(pObject);
        if (typeId == -1) {
            fprintf(stderr, "IBinSaver: trying to save unregistered object\n");
            abort();
        }
        DataChunk(&typeId, sizeof(typeId));
    }
}

IObjectBase* IBinSaver::LoadObject() {
    ui64 ptrId = 0;
    DataChunk(&ptrId, sizeof(ptrId));
    if (ptrId != 0) {
        if (!Objects.Get())
            Objects.Reset(new CObjectsHash);
        CObjectsHash::iterator pFound = Objects->find(ptrId);
        if (pFound != Objects->end())
            return pFound->second;
        int typeId;
        DataChunk(&typeId, sizeof(typeId));
        IObjectBase* pObj = pSaverClasses->CreateObject(typeId);
        Y_ASSERT(pObj != nullptr);
        if (pObj == nullptr) {
            fprintf(stderr, "IBinSaver: trying to load unregistered object\n");
            abort();
        }
        (*Objects)[ptrId] = pObj;
        ObjectQueue.push_back(pObj);
        return pObj;
    }
    return nullptr;
}

IBinSaver::~IBinSaver() {
    for (size_t i = 0; i < ObjectQueue.size(); ++i) {
        AddPolymorphicBase(1, ObjectQueue[i]);
    }
}
