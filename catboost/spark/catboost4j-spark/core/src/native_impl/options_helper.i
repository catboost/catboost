%{
#include <catboost/spark/catboost4j-spark/core/src/native_impl/options_helper.h>
#include <catboost/private/libs/options/catboost_options.h>
%}

%include "catboost_enums.i"

namespace NCatboostOptions {
    class TCatBoostOptions {
    public:
        explicit TCatBoostOptions(ETaskType taskType);
    };
}

%include "options_helper.h"