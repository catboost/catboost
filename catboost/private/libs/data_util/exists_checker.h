#pragma once

#include "path_with_scheme.h"

#include <library/cpp/object_factory/object_factory.h>

#include <util/system/fs.h>


namespace NCB {

    struct IExistsChecker {
        virtual bool Exists(const TPathWithScheme& pathWithScheme) const = 0;
        virtual bool IsSharedFs() const = 0;
        virtual ~IExistsChecker() = default;
    };

    using TExistsCheckerFactory = NObjectFactory::TParametrizedObjectFactory<IExistsChecker, TString>;


    /* creates scheme-dependent processor internally,
     * for heavy usage get IExistsChecker and call its Exists method directly
     */
    bool CheckExists(const TPathWithScheme& pathWithScheme);

    bool IsSharedFs(const TPathWithScheme& pathWithScheme);

    struct TFSExistsChecker : public IExistsChecker {
        bool Exists(const TPathWithScheme& pathWithScheme) const override {
            return NFs::Exists(pathWithScheme.Path);
        }
        bool IsSharedFs() const override {
            return false;
        }
    };

}
