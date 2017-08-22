#include "last_getopt.h"
#include "last_getopt_support.h"
#include "modchooser.h"
#include "opt.h"
#include "opt2.h"
#include "posix_getopt.h"
#include "ygetopt.h"

#include <library/svnversion/svnversion.h>
#include <library/build_info/build_info.h>


namespace NLastGetoptPrivate {

    TString InitVersionString() {
       TString ts = GetProgramSvnVersion();
       ts += "\n";
       ts += GetBuildInfo();
       return ts;
    }

    TString& VersionString();

    struct TInit {
        TInit() {
            VersionString() = InitVersionString();
        }
    } Init;

}
