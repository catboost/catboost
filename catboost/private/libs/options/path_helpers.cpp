#include <util/folder/path.h>


TString MakeAbsolutePath(const TString& path) {
    if (TFsPath(path).IsAbsolute()) {
        return path;
    }
    return JoinFsPaths(TFsPath::Cwd(), path);
}
