#include <util/generic/yexception.h>

#include <cerrno>
#include <util/system/info.h>

#if defined(_win_)
    #include <io.h>
#else
    #include <sys/wait.h>
    #include <unistd.h>
    #include <fcntl.h>
#endif

#include "daemon.h"

#ifdef _unix_
using namespace NDaemonMaker;

static bool Fork(EParent parent) {
    pid_t pid = fork();

    if (pid > 0) {
        int status = 0;
        while (waitpid(pid, &status, 0) < 0 && errno == EINTR) {
        }
        if (parent == callExitFromParent) {
            _exit(0);
        } else {
            return true;
        }
    } else if (pid < 0) {
        ythrow TSystemError() << "Cannot fork";
    }

    if (setsid() < 0) {
        ythrow TSystemError() << "Cannot setsid";
    }

    pid = fork();

    if (pid > 0) {
        _exit(0);
    } else if (pid < 0) {
        ythrow TSystemError() << "Cannot second fork";
    }
    return false;
}

#endif

static void CloseFromToExcept(int from, int to, const int* except) {
    (void)from;
    (void)to;
    (void)except;

#ifdef _unix_
    int mfd = NSystemInfo::MaxOpenFiles();
    for (int s = from; s < mfd && (to == -1 || s < to); s++) {
        for (const int* ex = except; *ex >= 0; ++ex) {
            if (s == *ex) {
                goto dontclose;
            }
        }
        while (close(s) == -1) {
            if (errno == EBADF) {
                break;
            }
            if (errno != EINTR) {
                ythrow TSystemError() << "close(" << s << ") failed";
            }
        }
    dontclose:;
    }
#endif /* _unix_ */
}

bool NDaemonMaker::MakeMeDaemon(ECloseDescriptors cd, EStdIoDescriptors iod, EChDir chd, EParent parent) {
    (void)cd;
    (void)iod;
    (void)chd;

#ifdef _unix_
    if (Fork(parent)) {
        return true;
    }

    if (chd == chdirRoot) {
        if (chdir("/")) {
            ythrow TSystemError() << "chdir(\"/\") failed";
        }
    }

    int fd[4] = {-1, -1, -1, -1};
    switch (iod) {
        case openYandexStd:
            fd[0] = open("yandex.stdin", O_RDONLY);
            if (fd[0] < 0) {
                ythrow TSystemError() << "Cannot open 'yandex.stdin'";
            }
            fd[1] = open("yandex.stdout", O_WRONLY | O_APPEND | O_CREAT, 660);
            if (fd[1] < 0) {
                ythrow TSystemError() << "Cannot open 'yandex.stdout'";
            }
            fd[2] = open("yandex.stderr", O_WRONLY | O_APPEND | O_CREAT, 660);
            if (fd[2] < 0) {
                ythrow TSystemError() << "Cannot open 'yandex.stderr'";
            }
            break;
        case openDevNull:
            fd[0] = open("/dev/null", O_RDWR, 0);
            break;
        case openNone:
            break;
        default:
            ythrow yexception() << "Unknown open descriptors mode: " << (int)iod;
    }

    const int except[4] = {
        fd[0],
        fd[1],
        fd[2],
        -1};
    if (closeAll == cd) {
        CloseFromToExcept(0, -1, except);
    } else if (closeStdIoOnly == cd) {
        CloseFromToExcept(0, 3, except);
    } else {
        ythrow yexception() << "Unknown close descriptors mode: " << (int)cd;
    }

    switch (iod) {
        case openYandexStd:
            /* Assuming that open(2) acquires fds in order. */
            dup2(fd[0], STDIN_FILENO);
            if (fd[0] > 2) {
                close(fd[0]);
            }
            dup2(fd[1], STDOUT_FILENO);
            if (fd[1] > 2) {
                close(fd[1]);
            }
            dup2(fd[2], STDERR_FILENO);
            if (fd[2] > 2) {
                close(fd[2]);
            }
            break;
        case openDevNull:
            dup2(fd[0], STDIN_FILENO);
            dup2(fd[0], STDOUT_FILENO);
            dup2(fd[0], STDERR_FILENO);
            if (fd[0] > 2) {
                close(fd[0]);
            }
            break;
        default:
            break;
    }
    return false;
#else
    return true;
#endif
}

void NDaemonMaker::CloseFrom(int fd) {
    static const int except[1] = {-1};
    CloseFromToExcept(fd, -1, except);
}
