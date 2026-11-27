#include "command_file_args.h"

#include <util/folder/path.h>
#include <util/generic/yexception.h>
#include <util/stream/file.h>

namespace NStage0Pycc {
    namespace {
        TFileInput OpenCommandFile(const TString& path) {
            try {
                return TFileInput{path};
            } catch (const yexception&) {
                if (!TFsPath{path}.Exists()) {
                    ythrow yexception() << "Command file '" << path << "' doesn't exist.";
                }
                throw;
            }
        }
    } // namespace

    TVector<TString> ReadCommandFileArgs(TConstArrayRef<const char*> arguments) {
        TVector<TString> result;
        for (const char* raw : arguments) {
            const TStringBuf argument{raw};
            if (argument == "--ya-start-command-file" || argument == "--ya-end-command-file") {
                continue;
            }
            if (!argument.StartsWith('@')) {
                result.emplace_back(argument);
                continue;
            }
            const TString path{argument.substr(1)};
            auto input = OpenCommandFile(path);
            TString line;
            while (input.ReadLine(line)) {
                if (line.find('\0') != TString::npos) {
                    ythrow yexception() << "Command file '" << path << "' contains a NUL byte.";
                }
                if (!line.empty() && line.back() == '\r') {
                    line.pop_back();
                }
                if (!line.empty()) {
                    result.push_back(line);
                }
            }
        }
        return result;
    }

} // namespace NStage0Pycc
