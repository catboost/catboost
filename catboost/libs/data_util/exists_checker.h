#pragma once

#include "path_with_scheme.h"

#include <library/object_factory/object_factory.h>


namespace NCB {

    struct IExistsChecker {
        virtual bool Exists(const TPathWithScheme& pathWithScheme) const = 0;
        virtual ~IExistsChecker() = default;
    };

    using TExistsCheckerFactory = NObjectFactory::TParametrizedObjectFactory<IExistsChecker, TString>;


    /* creates scheme-dependent processor internally,
     * for heavy usage get IExistsChecker and call its Exists method directly
     */
    bool CheckExists(const TPathWithScheme& pathWithScheme);

}
