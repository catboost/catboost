#include "modchooser.h"

#include <library/colorizer/colors.h>

#include <util/stream/output.h>
#include <util/stream/format.h>
#include <util/generic/yexception.h>
#include <util/generic/ptr.h>
#include <util/string/builder.h>
#include <util/string/join.h>
#include "last_getopt.h"

class PtrWrapper: public TMainClassV {
public:
    explicit PtrWrapper(const TMainFunctionPtr main)
        : Main(main)
    {
    }

    int operator()(const TVector<TString>& argv) override {
        const size_t argc = argv.size();
        TVector<const char*> ptrArgv(argc, nullptr);
        for (size_t i = 0; i < argc; ++i) {
            ptrArgv[i] = argv[i].c_str();
        }
        return Main(argc, &*ptrArgv.begin());
    }

private:
    TMainFunctionPtr Main;
};

class PtrvWrapper: public TMainClassV {
public:
    explicit PtrvWrapper(const TMainFunctionPtrV main)
        : Main(main)
    {
    }

    int operator()(const TVector<TString>& argv) override {
        return Main(argv);
    }

private:
    TMainFunctionPtrV Main;
};

class ClassWrapper: public TMainClassV {
public:
    explicit ClassWrapper(TMainClass* main)
        : Main(main)
    {
    }

    int operator()(const TVector<TString>& argv) override {
        const size_t argc = argv.size();
        TVector<const char*> ptrArgv(argc, nullptr);
        for (size_t i = 0; i < argc; ++i) {
            ptrArgv[i] = argv[i].c_str();
        }
        return (*Main)(argc, &*ptrArgv.begin());
    }

private:
    TMainClass* Main;
};

TModChooser::TMode::TMode(const TString& name, TMainClassV* main, const TString& descr)
    : Name(name)
    , Main(main)
    , Description(descr)
    , Separator(name ? false : true)
{
}

TModChooser::TModChooser()
    : ModesHelpOption("-?") // Default help option in last_getopt
    , VersionHandler(nullptr)
    , ShowSeparated(true)
    , SvnRevisionOptionDisabled(false)
{
}

TModChooser::~TModChooser() = default;

void TModChooser::AddMode(const TString& mode, const TMainFunctionRawPtr func, const TString& description) {
    AddMode(mode, TMainFunctionPtr(func), description);
}
void TModChooser::AddMode(const TString& mode, const TMainFunctionRawPtrV func, const TString& description) {
    AddMode(mode, TMainFunctionPtrV(func), description);
}

void TModChooser::AddMode(const TString& mode, const TMainFunctionPtr func, const TString& description) {
    Wrappers.push_back(new PtrWrapper(func));
    AddMode(mode, Wrappers.back().Get(), description);
}

void TModChooser::AddMode(const TString& mode, const TMainFunctionPtrV func, const TString& description) {
    Wrappers.push_back(new PtrvWrapper(func));
    AddMode(mode, Wrappers.back().Get(), description);
}

void TModChooser::AddMode(const TString& mode, TMainClass* func, const TString& description) {
    Wrappers.push_back(new ClassWrapper(func));
    AddMode(mode, Wrappers.back().Get(), description);
}

void TModChooser::AddMode(const TString& mode, TMainClassV* main, const TString& description) {
    if (Modes.FindPtr(mode)) {
        ythrow yexception() << "TMode '" << mode << "' already exists in TModChooser.";
    }

    Modes[mode] = UnsortedModes.emplace_back(new TMode(mode, main, description)).Get();
    return;
}

void TModChooser::AddGroupModeDescription(const TString& description) {
    UnsortedModes.push_back(new TMode(nullptr, nullptr, description.data()));
}

void TModChooser::AddAlias(const TString& alias, const TString& mode) {
    if (!Modes.FindPtr(mode)) {
        ythrow yexception() << "TMode '" << mode << "' not found in TModChooser.";
    }

    Modes[mode]->Aliases.push_back(alias);
    Modes[alias] = Modes[mode];
}

void TModChooser::SetDescription(const TString& descr) {
    Description = descr;
}

void TModChooser::SetModesHelpOption(const TString& helpOption) {
    ModesHelpOption = helpOption;
}

void TModChooser::SetVersionHandler(TVersionHandlerPtr handler) {
    VersionHandler = handler;
}

void TModChooser::SetSeparatedMode(bool separated) {
    ShowSeparated = separated;
}

void TModChooser::SetSeparationString(const TString& str) {
    SeparationString = str;
}

void TModChooser::DisableSvnRevisionOption() {
    SvnRevisionOptionDisabled = true;
}

int TModChooser::Run(const int argc, const char** argv) const {
    TVector<TString> args(argv, argv + argc);
    return Run(args);
}

int TModChooser::Run(const TVector<TString>& argv) const {
    Y_ENSURE(!argv.empty(), "Can't run TModChooser with empty list of arguments.");

    if (argv.size() == 1) {
        PrintHelp(argv[0]);
        return 0;
    }

    TString modeName = argv[1];
    if (modeName == "-h" || modeName == "--help" || modeName == "-?") {
        PrintHelp(argv[0]);
        return 0;
    }
    if (VersionHandler && (modeName == "-v" || modeName == "--version")) {
        VersionHandler();
        return 0;
    }
    if (!SvnRevisionOptionDisabled && modeName == "--svnrevision") {
        NLastGetopt::PrintVersionAndExit(nullptr);
    }

    TModes::const_iterator modeIter = Modes.find(modeName);
    if (modeIter == Modes.end()) {
        Cerr << "Unknown mode " << modeName.Quote() << "." << Endl;
        PrintHelp(argv[0]);
        return 1;
    }

    TVector<TString> nargv;
    nargv.reserve(argv.size() - 1);
    nargv.push_back(argv[0] + TString(" ") + modeIter->second->Name);

    for (size_t i = 2; i < argv.size(); ++i) {
        nargv.push_back(argv[i]);
    }
    return (*modeIter->second->Main)(nargv);
}

size_t TModChooser::TMode::CalculateFullNameLen() const {
    size_t len = Name.size();
    if (Aliases) {
        len += 2;
        for (auto& alias : Aliases) {
            len += alias.size() + 1;
        }
    }
    return len;
}

TString TModChooser::TMode::FormatFullName(size_t pad) const {
    TStringBuilder name;
    if (Aliases) {
        name << "{";
    }

    name << NColorizer::StdErr().GreenColor();
    name << Name;
    name << NColorizer::StdErr().OldColor();

    if (Aliases) {
        for (const auto& alias : Aliases) {
            name << "|" << NColorizer::StdErr().GreenColor() << alias << NColorizer::StdErr().OldColor();
        }
        name << "}";
    }

    auto len = CalculateFullNameLen();
    if (pad > len) {
        name << TString(" ") * (pad - len);
    }

    return name;
}

void TModChooser::PrintHelp(const TString& progName) const {
    Cerr << Description << Endl << Endl;
    Cerr << NColorizer::StdErr().BoldColor() << "Usage" << NColorizer::StdErr().OldColor() << ": " << progName << " MODE [MODE_OPTIONS]" << Endl;
    Cerr << Endl;
    Cerr << NColorizer::StdErr().BoldColor() << "Modes" << NColorizer::StdErr().OldColor() << ":" << Endl;
    size_t maxModeLen = 0;
    for (const auto& [name, mode] : Modes) {
        if (name != mode->Name)
            continue;  // this is an alias
        maxModeLen = Max(maxModeLen, mode->CalculateFullNameLen());
    }

    if (ShowSeparated) {
        for (const auto& unsortedMode : UnsortedModes)
            if (unsortedMode->Name.size()) {
                Cerr << "  " << unsortedMode->FormatFullName(maxModeLen + 4) << unsortedMode->Description << Endl;
            } else {
                Cerr << SeparationString << Endl;
                Cerr << unsortedMode->Description << Endl;
            }
    } else {
        for (const auto& mode : Modes) {
            if (mode.first != mode.second->Name)
                continue;  // this is an alias
            Cerr << "  " << mode.second->FormatFullName(maxModeLen + 4) << mode.second->Description << Endl;
        }
    }

    Cerr << Endl;
    Cerr << "To get help for specific mode type '" << progName << " MODE " << ModesHelpOption << "'" << Endl;
    if (VersionHandler)
        Cerr << "To print program version type '" << progName << " --version'" << Endl;
    if (!SvnRevisionOptionDisabled) {
        Cerr << "To print svn revision type --svnrevision" << Endl;
    }
    return;
}
