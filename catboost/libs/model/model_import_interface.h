#pragma once

#include "model.h"

#include <library/cpp/object_factory/object_factory.h>

#include <util/generic/buffer.h>
#include <util/stream/buffer.h>
#include <util/stream/file.h>
#include <util/system/fs.h>

namespace NCB {
    class IModelLoader {
    public:
        virtual TFullModel ReadModel(IInputStream* modelStream) const = 0;
        virtual TFullModel ReadModel(const TString& modelPath) const {
            CB_ENSURE(NFs::Exists(modelPath), "Model file doesn't exist: " << modelPath);
            TIFStream f(modelPath);
            return ReadModel(&f);
        }
        virtual TFullModel ReadModel(const void* data, size_t dataSize) const {
            TBuffer buf((char*)data, dataSize);
            TBufferInput bs(buf);
            return ReadModel(&bs);
        }
        virtual ~IModelLoader() = default;
    protected:
        void CheckModel(TFullModel* model) const;
    };

    using TModelLoaderFactory = NObjectFactory::TParametrizedObjectFactory<IModelLoader, EModelType>;

    extern void* BinaryModelLoaderRegistratorPointer;

    extern void* JsonModelLoaderRegistratorPointer;
}
