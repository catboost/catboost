#pragma once

#include <catboost/libs/model/enums.h>
#include <catboost/libs/model/model.h>

#include <util/generic/fwd.h>


// needed for forwarding exceptions from C++ to JVM
TFullModel ReadModelWrapper(
    const TString& modelFile,
    EModelType format = EModelType::CatboostBinary
) throw(yexception);
