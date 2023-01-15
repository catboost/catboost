#include "nice.h"

#include "platform.h"

#if defined(_unix_)
    #include <unistd.h>
#endif

bool Nice(int prioDelta) {
#if defined(_unix_)
    return nice(prioDelta) != -1;
#else
    return prioDelta == 0;
#endif
}
