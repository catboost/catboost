#include <build/scripts/c_templates/svnversion.h>
#include <util/stream/str.h>
#include <util/system/compiler.h>
#include <util/stream/output.h>

// ya make -DFORCE_VCS_INFO_UPDATE --vcs-file=<vcs.json> library/cpp/svnversion/test/
// ./library/svnversion/test/test[.exe]
int main() {
    Cout << "GetProgramSvnVersion(): " << GetProgramSvnVersion() << Endl;
    Cout << "GetCustomVersion(): " << GetCustomVersion() << Endl;
    Cout << "GetReleaseVersion(): " << GetReleaseVersion() << Endl;
    Cout << "PrintProgramSvnVersion(): " << Endl; PrintProgramSvnVersion();
    Cout << "GetArcadiaSourcePath(): " << GetArcadiaSourcePath() << Endl;
    Cout << "GetArcadiaSourceUrl(): " << GetArcadiaSourceUrl() << Endl;
    Cout << "GetArcadiaLastChange(): " << GetArcadiaLastChange() << Endl;
    Cout << "GetArcadiaLastChangeNum(): " << GetArcadiaLastChangeNum() << Endl;
    Cout << "GetArcadiaLastAuthor(): " << GetArcadiaLastAuthor() << Endl;
    Cout << "GetProgramSvnRevision(): " << GetProgramSvnRevision() << Endl;
    Cout << "GetProgramHash(): " << GetProgramHash() << Endl;
    Cout << "GetProgramCommitId(): " << GetProgramCommitId() << Endl;
    Cout << "GetProgramScmData(): " << GetProgramScmData() << Endl;
    Cout << "GetProgramBuildUser(): " << GetProgramBuildUser() << Endl;
    Cout << "GetProgramBuildHost(): " << GetProgramBuildHost() << Endl;
    Cout << "GetProgramBuildDate(): " << GetProgramBuildDate() << Endl;
    Cout << "GetVCS(): " << GetVCS() << Endl;
    Cout << "GetBranch(): " << GetBranch() << Endl;
    Cout << "GetTag(): " << GetTag() << Endl;
    Cout << "GetArcadiaPatchNumber(): " << GetArcadiaPatchNumber() << Endl;
    Cout << "GetVCSDirty(): " << GetVCSDirty() << Endl;
    return 0;
}

