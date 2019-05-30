#pragma once

#include <catboost/libs/options/enums.h>
#include <util/system/types.h>
#include <util/generic/ptr.h>
#include <util/stream/input.h>

namespace NCB {

    //this one will be in model and independent from out data
    class IFeatureCalcer : public TThrRefBase {
    public:

        virtual EFeatureEstimatorType Type() const = 0;

        virtual size_t FeatureCount() const = 0;

        //TODO: (noxoomo, kirillovs@): remove duplication with ICtrProvider
        virtual bool IsSerializable() const {
            return false;
        }

        virtual void Save(IOutputStream* ) const {
            Y_FAIL("Serialization not allowed");
        };

        virtual void Load(IInputStream* ) {
            Y_FAIL("Deserialization not allowed");
        };

        //TBD: serializations and apply
    };
}
