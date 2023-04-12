// Used indirectly. See build/scripts/vcs_info.py
// ya-bin dump vcs-info > vcs.json
// python build/scripts/vcs_info.py vcs.json out.c build/scripts/c_templates/svn_interface.c <any_prefix>library/cpp/svnversion<any_suffix>


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
