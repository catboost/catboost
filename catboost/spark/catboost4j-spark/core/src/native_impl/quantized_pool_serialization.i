%{
#include <catboost/private/libs/quantized_pool/serialization.h>
%}

%include <bindings/swiglib/stroka.swg>

%include "data_provider.i"


%catches(yexception) NCB::SaveQuantizedPool(const TDataProviderPtr& dataProvider, TString fileName);

namespace NCB {
    void SaveQuantizedPool(const TDataProviderPtr& dataProvider, TString fileName);
}
