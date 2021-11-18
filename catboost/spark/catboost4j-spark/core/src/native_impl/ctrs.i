%{
#include <catboost/libs/data/ctrs.h>
#include <catboost/spark/catboost4j-spark/core/src/native_impl/ctrs.h>
%}

%include "data_provider.i"
%include "string.i"

namespace NCB {
    struct TPrecomputedOnlineCtrMetaData {
    public:
        void Append(TPrecomputedOnlineCtrMetaData& add);

        // Use JSON as string to be able to use in JVM binding as well
        TString SerializeToJson() const;
        static TPrecomputedOnlineCtrMetaData DeserializeFromJson(
            const TString& serializedJson
        );
    };
}

%include "ctrs.h"
