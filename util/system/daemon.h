#pragma once

namespace NDaemonMaker {
    enum ECloseDescriptors {
        closeAll = 0,
        closeStdIoOnly
    };

    enum EStdIoDescriptors {
        openNone = 0,
        openDevNull,
        openYandexStd
    };

    enum EChDir {
        chdirNone = 0,
        chdirRoot
    };

    enum EParent {
        callExitFromParent = 0,
        returnFromParent
    };

    bool MakeMeDaemon(ECloseDescriptors cd = closeAll, EStdIoDescriptors iod = openDevNull, EChDir chd = chdirRoot, EParent parent = callExitFromParent);
    void CloseFrom(int fd);
}
