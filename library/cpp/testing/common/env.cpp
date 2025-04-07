#include "env.h"

#include <build/scripts/c_templates/svnversion.h>

#include <util/folder/dirut.h>
#include <util/folder/path.h>
#include <util/generic/singleton.h>
#include <util/stream/file.h>
#include <util/stream/fwd.h>
#include <util/system/env.h>
#include <util/system/file.h>
#include <util/system/file_lock.h>
#include <util/system/guard.h>

#include <library/cpp/json/json_reader.h>
#include <library/cpp/json/json_value.h>
#include <library/cpp/json/json_writer.h>

TString ArcadiaSourceRoot() {
    if (const auto& sourceRoot = NPrivate::GetTestEnv().SourceRoot) {
        return sourceRoot;
    } else {
        return GetArcadiaSourcePath();
    }
}

TString BuildRoot() {
    if (const auto& buildRoot = NPrivate::GetTestEnv().BuildRoot) {
        return buildRoot;
    } else {
        return GetArcadiaSourcePath();
    }
}

TString ArcadiaFromCurrentLocation(TStringBuf where, TStringBuf path) {
    return (TFsPath(ArcadiaSourceRoot()) / TFsPath(where).Parent() / path).Fix();
}

TString BinaryPath(TStringBuf path) {
    return (TFsPath(BuildRoot()) / path).Fix();
}

TString GetArcadiaTestsData() {
    return ArcadiaSourceRoot() + "/atd_ro_snapshot";
}

TString GetWorkPath() {
    TString workPath = NPrivate::GetTestEnv().WorkPath;
    if (workPath) {
        return workPath;
    }

    return NPrivate::GetCwd();
}

TFsPath GetOutputPath() {
    return GetWorkPath() + "/testing_out_stuff";
}

const TString& GetRamDrivePath() {
    return NPrivate::GetTestEnv().RamDrivePath;
}

const TString& GetYtHddPath() {
    return NPrivate::GetTestEnv().YtHddPath;
}

const TString& GetOutputRamDrivePath() {
    return NPrivate::GetTestEnv().TestOutputRamDrivePath;
}

const TString& GdbPath() {
    return NPrivate::GetTestEnv().GdbPath;
}

const TString& GetTestParam(TStringBuf name) {
    const static TString def = "";
    return GetTestParam(name, def);
}

const TString& GetTestParam(TStringBuf name, const TString& def) {
    auto& testParameters = NPrivate::GetTestEnv().TestParameters;
    auto it = testParameters.find(name.data());
    if (it != testParameters.end()) {
        return it->second;
    }
    return def;
}

const TString& GetGlobalResource(TStringBuf name) {
    auto& resources = NPrivate::GetTestEnv().GlobalResources;
    auto it = resources.find(name.data());
    Y_ABORT_UNLESS(it != resources.end());
    return it->second;
}

void AddEntryToCoreSearchFile(const TString& filename, TStringBuf cmd, int pid, const TFsPath& binaryPath = TFsPath(), const TFsPath& cwd = TFsPath()) {
    auto lock = TFileLock(filename);
    TGuard<TFileLock> guard(lock);

    TOFStream output(TFile(filename, WrOnly | ForAppend | OpenAlways));

    NJson::TJsonWriter writer(&output, false);
    writer.OpenMap();
    writer.Write("cmd", cmd);
    writer.Write("pid", pid);
    if (binaryPath) {
        writer.Write("binary_path", binaryPath);
    }
    if (cwd) {
        writer.Write("cwd", cwd);
    }
    writer.CloseMap();
    writer.Flush();

    output.Write("\n");
}

void WatchProcessCore(int pid, const TFsPath& binaryPath, const TFsPath& cwd) {
    auto& filename = NPrivate::GetTestEnv().CoreSearchFile;
    if (filename) {
        AddEntryToCoreSearchFile(filename, "add", pid, binaryPath, cwd);
    }
}

void StopProcessCoreWatching(int pid) {
    auto& filename = NPrivate::GetTestEnv().CoreSearchFile;
    if (filename) {
        AddEntryToCoreSearchFile(filename, "drop", pid);
    }
}

bool FromYaTest() {
    return NPrivate::GetTestEnv().IsRunningFromTest;
}

namespace NPrivate {
    TTestEnv::TTestEnv() {
        ReInitialize();
    }

    void TTestEnv::ReInitialize() {
        IsRunningFromTest = false;
        SourceRoot = "";
        BuildRoot = "";
        WorkPath = "";
        RamDrivePath = "";
        YtHddPath = "";
        TestOutputRamDrivePath = "";
        GdbPath = "";
        CoreSearchFile = "";
        EnvFile = "";
        TestParameters.clear();
        GlobalResources.clear();

        const TString contextFilename = GetEnv("YA_TEST_CONTEXT_FILE");
        if (contextFilename && TFsPath(contextFilename).Exists()) {
            NJson::TJsonValue context;
            NJson::ReadJsonTree(TFileInput(contextFilename).ReadAll(), &context);

            NJson::TJsonValue* value;

            value = context.GetValueByPath("runtime.source_root");
            if (value) {
                SourceRoot = value->GetStringSafe("");
            }

            value = context.GetValueByPath("runtime.build_root");
            if (value) {
                BuildRoot = value->GetStringSafe("");
            }

            value = context.GetValueByPath("runtime.work_path");
            if (value) {
                WorkPath = value->GetStringSafe("");
            }

            value = context.GetValueByPath("runtime.ram_drive_path");
            if (value) {
                RamDrivePath = value->GetStringSafe("");
            }

            value = context.GetValueByPath("runtime.yt_hdd_path");
            if (value) {
                YtHddPath = value->GetStringSafe("");
            }

            value = context.GetValueByPath("runtime.test_output_ram_drive_path");
            if (value) {
                TestOutputRamDrivePath = value->GetStringSafe("");
            }

            value = context.GetValueByPath("runtime.gdb_bin");
            if (value) {
                GdbPath = value->GetStringSafe("");
            }

            value = context.GetValueByPath("runtime.test_params");
            if (value) {
                for (const auto& entry : context.GetValueByPath("runtime.test_params")->GetMap()) {
                    TestParameters[entry.first] = entry.second.GetStringSafe("");
                }
            }

            value = context.GetValueByPath("resources.global");
            if (value) {
                for (const auto& entry : value->GetMap()) {
                    GlobalResources[entry.first] = entry.second.GetStringSafe("");
                }
            }

            value = context.GetValueByPath("internal.core_search_file");
            if (value) {
                CoreSearchFile = value->GetStringSafe("");
            }

            value = context.GetValueByPath("internal.env_file");
            if (value) {
                EnvFile = value->GetStringSafe("");
                if (TFsPath(EnvFile).Exists()) {
                    TFileInput file(EnvFile);
                    NJson::TJsonValue envVar;
                    TString ljson;
                    while (file.ReadLine(ljson) > 0) {
                        NJson::ReadJsonTree(ljson, &envVar);
                        for (const auto& entry : envVar.GetMap()) {
                            SetEnv(entry.first, entry.second.GetStringSafe(""));
                        }
                    }
                }
            }
        }

        if (!YtHddPath) {
            YtHddPath = GetEnv("HDD_PATH");
        }

        if (!SourceRoot) {
            SourceRoot = GetEnv("ARCADIA_SOURCE_ROOT");
        }

        if (!BuildRoot) {
            BuildRoot = GetEnv("ARCADIA_BUILD_ROOT");
        }

        if (!WorkPath) {
            WorkPath = GetEnv("TEST_WORK_PATH");
        }

        if (!RamDrivePath) {
            RamDrivePath = GetEnv("YA_TEST_RAM_DRIVE_PATH");
        }

        if (!TestOutputRamDrivePath) {
            TestOutputRamDrivePath = GetEnv("YA_TEST_OUTPUT_RAM_DRIVE_PATH");
        }

        const TString fromEnv = GetEnv("YA_TEST_RUNNER");
        IsRunningFromTest = (fromEnv == "1");
    }

    void TTestEnv::AddTestParam(TStringBuf name, TStringBuf value) {
        TestParameters[TString{name}] = value;
    }

    TString GetCwd() {
        try {
            return NFs::CurrentWorkingDirectory();
        } catch (...) {
            return {};
        }
    }

    const TTestEnv& GetTestEnv() {
        return *Singleton<TTestEnv>();
    }
}
