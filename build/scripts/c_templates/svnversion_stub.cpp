// svnversion_stub.cpp — stub implementation for Bazel builds.
// The ya/CMake build generates VCS info at build time; Bazel uses this stub.
#include "svnversion.h"
#include <cstdio>
#include <cstdlib>

extern "C" {
const char* GetProgramSvnVersion()    { return "Bazel build (no VCS info)"; }
void        PrintProgramSvnVersion()  { puts(GetProgramSvnVersion()); }
const char* GetArcadiaSourcePath()    { return ""; }
const char* GetArcadiaSourceUrl()     { return ""; }
const char* GetArcadiaLastChange()    { return "0"; }
int         GetArcadiaLastChangeNum() { return 0; }
const char* GetArcadiaLastAuthor()    { return ""; }
int         GetProgramSvnRevision()   { return 0; }
const char* GetVCSDirty()             { return ""; }
const char* GetProgramHash()          { return ""; }
const char* GetProgramCommitId()      { return ""; }
void PrintSvnVersionAndExit(int, char*[])          { exit(0); }
void PrintSvnVersionAndExitEx(int, char*[], const char*) { exit(0); }
void PrintSvnVersionAndExit0()        { exit(0); }
const char* GetCustomVersion()        { return ""; }
const char* GetReleaseVersion()       { return ""; }
const char* GetProgramScmData()       { return ""; }
const char* GetProgramShortVersionData() { return ""; }
const char* GetProgramBuildUser()     { return ""; }
const char* GetProgramBuildHost()     { return ""; }
const char* GetProgramBuildDate()     { return ""; }
int         GetProgramBuildTimestamp(){ return 0; }
const char* GetVCS()                  { return ""; }
const char* GetBranch()               { return ""; }
const char* GetTag()                  { return ""; }
int         GetArcadiaPatchNumber()   { return 0; }
}
