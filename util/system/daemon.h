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

    bool MakeMeDaemon(ECloseDescriptors cd = closeAll, EStdIoDescriptors iod = openDevNull, EChDir chd = chdirRoot, bool exitFromParent = true);
    void CloseFrom(int fd);
}
