
%{
#include <catboost/libs/data/data_provider.h>
#include <catboost/libs/data/objects.h>
%}

namespace NCB {
    class TRawObjectsDataProviderPtr;
    class TQuantizedObjectsDataProviderPtr {
    };


    /**
     * Black box for SWIG because it is impossible to declare it using TIntrusivePtr because
     *  TDataProver does not have a default constructor
     *  All methods will be added as external.
     */
    class TDataProviderPtr {
    public:
        %extend {
            i64 GetObjectCount() const {
                return (*self)->GetObjectCount();
            }
        }
    };
}
