%{
#include <catboost/libs/helpers/exception.h>

#include <library/cpp/threading/local_executor/local_executor.h>

#include <util/generic/cast.h>
#include <util/system/info.h>

%}

%include "defaults.i"


%catches(std::exception) NPar::TLocalExecutor::Init(i32 threadCount);

namespace NPar {

    class TLocalExecutor {
    public:
        %extend {
            // threadCount can be positive or equal to -1 (in this case the number of CPU cores is used)
            void Init(i32 threadCount) {
                if (threadCount == -1) {
                    threadCount = SafeIntegerCast<i32>(NSystemInfo::CachedNumberOfCpus());
                } else {
                    CB_ENSURE(threadCount > 0, "Thread count must be positive or -1");
                }
                self->RunAdditionalThreads(threadCount - 1);
            }
        }
    };
    
}