#include "last_getopt.h"
#include "last_getopt_support.h"
#include "modchooser.h"
#include "opt.h"
#include "opt2.h"
#include "posix_getopt.h"
#include "ygetopt.h"

#if defined(YMAKE)
#include <library/svnversion/svnversion.h>
#else
#include <util/string/builder.h>
#endif


namespace NLastGetoptPrivate {

    TString InitVersionString() {
#ifdef YMAKE
       return GetProgramSvnVersion();
#else
       TStringBuilder builder;
#if defined(PROGRAM_VERSION)
        builder << PROGRAM_VERSION << Endl;
#elif defined(SVN_REVISION)
        builder << "revision: " << SVN_REVISION << " from " << SVN_ARCROOT << " at " << SVN_TIME << Endl;
#else
        builder << "program version: not implemented" << Endl;
#endif
        return builder;
#endif
    }

    TString& VersionString();

    struct TInit {
        TInit() {
            VersionString() = InitVersionString();
        }
    } Init;

}
