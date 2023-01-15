#pragma once

#include "implementation_type_enum.h"

#include <library/cpp/getopt/small/last_getopt_opts.h>
#include <library/cpp/object_factory/object_factory.h>

namespace NCB {
    class IModeNormalizeModelImplementation {
    public:
        virtual int mode_normalize_model(int args, const char** argv) const = 0;
        virtual ~IModeNormalizeModelImplementation() = default;
    };

    using TModeNormalizeModelImplementationFactory = NObjectFactory::TParametrizedObjectFactory<IModeNormalizeModelImplementation, EImplementationType>;
}
