#include "dirut.h"
#include "filelist.h"
#include "iterator.h"

#include <util/system/defaults.h>

void TFileEntitiesList::Fill(const TString& dirname, TStringBuf prefix, TStringBuf suffix, int depth, bool sort) {
    TDirIterator::TOptions opts;
    opts.SetMaxLevel(depth);
    if (sort) {
        opts.SetSortByName();
    }

    TDirIterator dir(dirname, opts);
    Clear();

    size_t dirNameLength = dirname.length();
    while (dirNameLength && (dirname[dirNameLength - 1] == '\\' || dirname[dirNameLength - 1] == '/')) {
        --dirNameLength;
    }

    for (auto file = dir.begin(); file != dir.end(); ++file) {
        if (file->fts_pathlen == file->fts_namelen || file->fts_pathlen <= dirNameLength) {
            continue;
        }

        TStringBuf filename = file->fts_path + dirNameLength + 1;

        if (filename.empty() || !filename.StartsWith(prefix) || !filename.EndsWith(suffix)) {
            continue;
        }

        if (((Mask & EM_FILES) && file->fts_info == FTS_F) || ((Mask & EM_DIRS) && file->fts_info == FTS_D) || ((Mask & EM_SLINKS) && file->fts_info == FTS_SL)) {
            ++FileNamesSize;
            FileNames.Append(filename.data(), filename.size() + 1);
        }
    }

    Restart();
}
