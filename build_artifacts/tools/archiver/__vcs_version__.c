#define ARCADIA_PATCH_NUMBER 47960
#define ARCADIA_SOURCE_HG_HASH "e736fa8da89925cb98ba85d7d6369626a59ea55f"
#define ARCADIA_SOURCE_LAST_AUTHOR "makar <kulishenko.m@mail.ru>"
#define ARCADIA_SOURCE_LAST_CHANGE 9293085
#define ARCADIA_SOURCE_PATH "/Users/makar/Documents/Course work/Repository/catboost"
#define ARCADIA_SOURCE_REVISION -1
#define ARCADIA_SOURCE_URL ""
#define ARCADIA_TAG ""
#define BRANCH "heads/develop"
#define BUILD_DATE "2025-02-18T18:10:48.000000Z"
#define BUILD_TIMESTAMP 1739902248
#define BUILD_USER "makar"
#define CUSTOM_VERSION ""
#define DIRTY ""
#define PROGRAM_VERSION "Git info:\n"\
    "    Commit: e736fa8da89925cb98ba85d7d6369626a59ea55f\n"\
    "    Branch: heads/develop\n"\
    "    Author: makar <kulishenko.m@mail.ru>\n"\
    "    Summary: fix CatBoostEstimator class\n"\
    "    git-svn info:\n"\
    "    Last Changed Rev: 9293085\n"\
    "\n"\
    "Other info:\n"\
    "    Build by: makar\n"\
    "    Top src dir: /Users/makar/Documents/Course work/Repository/catboost\n"\
    ""
#define SCM_DATA "Git info:\n"\
    "    Commit: e736fa8da89925cb98ba85d7d6369626a59ea55f\n"\
    "    Branch: heads/develop\n"\
    "    Author: makar <kulishenko.m@mail.ru>\n"\
    "    Summary: fix CatBoostEstimator class\n"\
    "    git-svn info:\n"\
    "    Last Changed Rev: 9293085\n"\
    ""
#define VCS "git"
// Used indirectly. See build/scripts/vcs_info.py
// ya-bin dump vcs-info > vcs.json
// python build/scripts/vcs_info.py vcs.json out.c build/scripts/c_templates/svn_interface.c <any_prefix>library/cpp/svnversion<any_suffix>
// DO_NOT_STYLE because this script is sometimes implicilty added as a styling target to STYLE_CPP

#include "build/scripts/c_templates/svnversion.h"

#define STR1(x) #x
#define STR2(x) STR1(x)

const char* GetProgramSvnVersion() {
#if defined(REVISION)
    // for package systems generating from svn export but providing REVISION macro
    #define STRREVISION STR2(REVISION)
    #define REVISIONINFO "r" STRREVISION
    #if defined(PROGRAM_VERSION)
    return PROGRAM_VERSION "\n\n" REVISIONINFO;
    #else
    return REVISIONINFO " "__DATE__
                        " "__TIME__;
    #endif
#elif defined(PROGRAM_VERSION)
    return PROGRAM_VERSION;
#else
    return "No program version found";
#endif
}

const char* GetArcadiaSourcePath() {
#if defined(ARCADIA_SOURCE_PATH)
    return ARCADIA_SOURCE_PATH;
#else
    return "";
#endif
}

const char* GetArcadiaSourceUrl() {
#if defined(ARCADIA_SOURCE_URL)
    return ARCADIA_SOURCE_URL;
#else
    return "";
#endif
}

int GetArcadiaLastChangeNum() {
#if defined(ARCADIA_SOURCE_LAST_CHANGE)
    return ARCADIA_SOURCE_LAST_CHANGE;
#else
    return 0;
#endif
}

const char* GetArcadiaLastChange() {
#if defined(ARCADIA_SOURCE_LAST_CHANGE)
    return STR2(ARCADIA_SOURCE_LAST_CHANGE);
#else
    return "";
#endif
}

const char* GetArcadiaLastAuthor() {
#if defined(ARCADIA_SOURCE_LAST_AUTHOR)
    return ARCADIA_SOURCE_LAST_AUTHOR;
#else
    return "";
#endif
}

int GetProgramSvnRevision() {
#if defined(ARCADIA_SOURCE_REVISION)
    return ARCADIA_SOURCE_REVISION;
#else
    return 0;
#endif
}

const char* GetVCSDirty()
{
#if defined(DIRTY)
    return DIRTY;
#else
    return 0;
#endif
}

const char* GetProgramHash() {
#if defined(ARCADIA_SOURCE_HG_HASH)
    return ARCADIA_SOURCE_HG_HASH;
#else
    return "";
#endif
}

const char* GetProgramCommitId() {
#if defined(ARCADIA_SOURCE_REVISION)
    if (ARCADIA_SOURCE_REVISION <= 0) {
        return GetProgramHash();
    }
    return STR2(ARCADIA_SOURCE_REVISION);
#else
    return GetProgramHash();
#endif
}

const char* GetProgramScmData() {
#if defined(SCM_DATA)
    return SCM_DATA;
#else
    return "";
#endif
}

const char* GetProgramShortVersionData() {
#if defined(SVN_REVISION) && defined(SVN_TIME)
    return STR2(SVN_REVISION) " (" SVN_TIME ")";
#else
    return GetProgramHash();
#endif
}

const char* GetProgramBuildUser() {
#if defined(BUILD_USER)
    return BUILD_USER;
#else
    return "";
#endif
}

const char* GetProgramBuildHost() {
#if defined(BUILD_HOST)
    return BUILD_HOST;
#else
    return "";
#endif
}

const char* GetProgramBuildDate() {
#if defined(BUILD_DATE)
    return BUILD_DATE;
#else
    return "";
#endif
}

const char* GetCustomVersion() {
#if defined(CUSTOM_VERSION)
    return CUSTOM_VERSION;
#else
    return "";
#endif
}

const char* GetReleaseVersion() {
#if defined(RELEASE_VERSION)
    return RELEASE_VERSION;
#else
    return "";
#endif
}

int GetProgramBuildTimestamp() {
#if defined(BUILD_TIMESTAMP)
    return BUILD_TIMESTAMP;
#else
    return 0;
#endif
}

const char* GetVCS() {
#if defined(VCS)
    return VCS;
#else
    return "";
#endif
}

const char* GetBranch() {
#if defined(BRANCH)
    return BRANCH;
#else
    return "";
#endif
}

int GetArcadiaPatchNumber() {
#if defined(ARCADIA_PATCH_NUMBER)
    return ARCADIA_PATCH_NUMBER;
#else
    return 42;
#endif
}

const char* GetTag() {
#if defined(ARCADIA_TAG)
    return ARCADIA_TAG;
#else
    return "";
#endif
}
