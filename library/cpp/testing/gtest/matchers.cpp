#include <library/cpp/testing/gtest/matchers.h>

#include <library/cpp/testing/common/env.h>

#include <util/folder/path.h>
#include <util/stream/file.h>
#include <util/system/fs.h>

bool NGTest::NDetail::MatchOrUpdateGolden(std::string_view actualContent, const TString& goldenFilename) {
    if (!GetTestParam("GTEST_UPDATE_GOLDEN").empty()) {
        Y_ENSURE(NFs::MakeDirectoryRecursive(TFsPath(goldenFilename).Parent()));
        TFile file(goldenFilename, CreateAlways);
        file.Write(actualContent.data(), actualContent.size());
        Cerr << "The data[" << actualContent.size() << "] has written to golden file " << goldenFilename << Endl;
        return true;
    }

    if (!NFs::Exists(goldenFilename)) {
        return actualContent.empty();
    }
    TFile goldenFile(goldenFilename, RdOnly);
    return actualContent == TFileInput(goldenFile).ReadAll();
}
