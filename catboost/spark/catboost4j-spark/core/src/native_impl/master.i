%{
#include <catboost/spark/catboost4j-spark/core/src/native_impl/master.h>
%}

%include <bindings/swiglib/stroka.swg>

%include "tvector.i"


%catches(std::exception) ModeFitImpl(const TVector<TString>& args);
%catches(std::exception) ShutdownWorkers(const TString& hostsFile, i32 timeoutInSeconds);

%include "master.h"