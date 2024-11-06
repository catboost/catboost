#include "exit.h"

namespace NYT {

////////////////////////////////////////////////////////////////////////////////

void AbortProcessSilently(int exitCode)
{
    _exit(exitCode);
}

void AbortProcessDramatically(int exitCode, TStringBuf exitCodeStr, TStringBuf message)
{
    fprintf(stderr, "\n");
    if (message) {
        fprintf(stderr, "*** %s\n", message.data());
    }
    fprintf(stderr, "*** Aborting process with exit code %d", exitCode);
    if (exitCodeStr) {
        fprintf(stderr, " (%s)", exitCodeStr.data());
    }
    fprintf(stderr, "\n");
    _exit(exitCode);
}

////////////////////////////////////////////////////////////////////////////////

} // namespace NYT
