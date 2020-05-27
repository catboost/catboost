#include "last_getopt.h"
#include "last_getopt_support.h"
#include "modchooser.h"
#include "opt.h"
#include "opt2.h"
#include "posix_getopt.h"
#include "ygetopt.h"

#include <library/cpp/svnversion/svnversion.h>
#include <library/cpp/build_info/build_info.h>

namespace NLastGetoptPrivate {
    TString InitVersionString() {
        TString ts = GetProgramSvnVersion();
        ts += "\n";
        ts += GetBuildInfo();
        TString sandboxTaskId = GetSandboxTaskId();
        if (sandboxTaskId != TString("0")) {
            ts += "\nSandbox task id: ";
            ts += sandboxTaskId;
        }
        return ts;
    }

    TString InitShortVersionString() {
        TString ts = GetProgramShortVersionData();
        return ts;
    }

    TString& VersionString();
    TString& ShortVersionString();

    struct TInit {
        TInit() {
            VersionString() = InitVersionString();
            ShortVersionString() = InitShortVersionString();
        }
    } Init;

}
