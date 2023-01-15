#include "thread_helper.h"

#include <util/generic/singleton.h>

TMtpQueueHelper& TMtpQueueHelper::Instance() {
    return *Singleton<TMtpQueueHelper>();
}
