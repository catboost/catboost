#include "completer.h"
#include "completer_command.h"
#include "completion_generator.h"
#include "last_getopt.h"
#include "modchooser.h"

#include <library/cpp/colorizer/colors.h>

#include <util/stream/output.h>
#include <util/stream/format.h>
#include <util/generic/yexception.h>
#include <util/generic/ptr.h>
#include <util/string/builder.h>
#include <util/string/join.h>

class PtrWrapper: public TMainClass {
public:
    explicit PtrWrapper(const TMainFunctionPtr main)
        : Main(main)
    {
    }

    int operator()(const int argc, const char** argv) override {
        return Main(argc, argv);
    }

private:
    TMainFunctionPtr Main;
};

class PtrvWrapper: public TMainClass {
public:
    explicit PtrvWrapper(const TMainFunctionPtrV main)
        : Main(main)
    {
    }

    int operator()(const int argc, const char** argv) override {
        TVector<TString> nargv(argv, argv + argc);
        return Main(nargv);
    }

private:
    TMainFunctionPtrV Main;
};

class ClassWrapper: public TMainClass {
public:
    explicit ClassWrapper(TMainClassV* main)
        : Main(main)
    {
    }

    int operator()(const int argc, const char** argv) override {
        TVector<TString> nargv(argv, argv + argc);
        return (*Main)(nargv);
    }

private:
    TMainClassV* Main;
};

TModChooser::TMode::TMode(const TString& name, TMainClass* main, const TString& descr, bool hidden, bool noCompletion)
    : Name(name)
    , Main(main)
    , Description(descr)
    , Hidden(hidden)
    , NoCompletion(noCompletion)
{
}

TModChooser::TModChooser()
    : ModesHelpOption("-?") // Default help option in last_getopt
    , VersionHandler(nullptr)
    , ShowSeparated(true)
    , SvnRevisionOptionDisabled(false)
    , PrintShortCommandInUsage(false)
{
}

TModChooser::~TModChooser() = default;

void TModChooser::AddMode(const TString& mode, const TMainFunctionRawPtr func, const TString& description, bool hidden, bool noCompletion) {
    AddMode(mode, TMainFunctionPtr(func), description, hidden, noCompletion);
}

void TModChooser::AddMode(const TString& mode, const TMainFunctionRawPtrV func, const TString& description, bool hidden, bool noCompletion) {
    AddMode(mode, TMainFunctionPtrV(func), description, hidden, noCompletion);
}

void TModChooser::AddMode(const TString& mode, const TMainFunctionPtr func, const TString& description, bool hidden, bool noCompletion) {
    Wrappers.push_back(new PtrWrapper(func));
    AddMode(mode, Wrappers.back().Get(), description, hidden, noCompletion);
}

void TModChooser::AddMode(const TString& mode, const TMainFunctionPtrV func, const TString& description, bool hidden, bool noCompletion) {
    Wrappers.push_back(new PtrvWrapper(func));
    AddMode(mode, Wrappers.back().Get(), description, hidden, noCompletion);
}

void TModChooser::AddMode(const TString& mode, TMainClass* func, const TString& description, bool hidden, bool noCompletion) {
    if (Modes.FindPtr(mode)) {
        ythrow yexception() << "TMode '" << mode << "' already exists in TModChooser.";
    }

    Modes[mode] = UnsortedModes.emplace_back(new TMode(mode, func, description, hidden, noCompletion)).Get();
}

void TModChooser::AddMode(const TString& mode, TMainClassV* func, const TString& description, bool hidden, bool noCompletion) {
    Wrappers.push_back(new ClassWrapper(func));
    AddMode(mode, Wrappers.back().Get(), description, hidden, noCompletion);
}

void TModChooser::AddGroupModeDescription(const TString& description, bool hidden, bool noCompletion) {
    UnsortedModes.push_back(new TMode(nullptr, nullptr, description.data(), hidden, noCompletion));
}

void TModChooser::SetDefaultMode(const TString& mode) {
    DefaultMode = mode;
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

void TModChooser::SetPrintShortCommandInUsage(bool printShortCommandInUsage = false) {
    PrintShortCommandInUsage = printShortCommandInUsage;
}

void TModChooser::DisableSvnRevisionOption() {
    SvnRevisionOptionDisabled = true;
}

void TModChooser::AddCompletions(TString progName, const TString& name, bool hidden, bool noCompletion) {
    if (CompletionsGenerator == nullptr) {
        CompletionsGenerator = NLastGetopt::MakeCompletionMod(this, std::move(progName), name);
        AddMode(name, CompletionsGenerator.Get(), "generate autocompletion files", hidden, noCompletion);
    }
}

int TModChooser::Run(const int argc, const char** argv) const {
    Y_ENSURE(argc, "Can't run TModChooser with empty list of arguments.");

    bool shiftArgs = true;
    TString modeName;
    if (argc == 1) {
        if (DefaultMode.empty()) {
            PrintHelp(argv[0]);
            return 0;
        } else {
            modeName = DefaultMode;
            shiftArgs = false;
        }
    } else {
        modeName = argv[1];
    }

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

    auto modeIter = Modes.find(modeName);
    if (modeIter == Modes.end() && !DefaultMode.empty()) {
        modeIter = Modes.find(DefaultMode);
        shiftArgs = false;
    }

    if (modeIter == Modes.end()) {
        Cerr << "Unknown mode " << modeName.Quote() << "." << Endl;
        PrintHelp(argv[0]);
        return 1;
    }

    if (shiftArgs) {
        TString firstArg;
        TVector<const char*> nargv(Reserve(argc - 1));

        if (PrintShortCommandInUsage) {
            firstArg = modeIter->second->Name;
        } else {
            firstArg = argv[0] + TString(" ") + modeIter->second->Name;
        }

        nargv.push_back(firstArg.data());

        for (int i = 2; i < argc; ++i) {
            nargv.push_back(argv[i]);
        }

        return (*modeIter->second->Main)(nargv.size(), nargv.data());
    } else {
        return (*modeIter->second->Main)(argc, argv);
    }
}

int TModChooser::Run(const TVector<TString>& argv) const {
    TVector<const char*> nargv(Reserve(argv.size()));
    for (auto& arg : argv) {
        nargv.push_back(arg.c_str());
    }
    return Run(nargv.size(), nargv.data());
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
            if (!unsortedMode->Hidden) {
                if (unsortedMode->Name.size()) {
                    Cerr << "  " << unsortedMode->FormatFullName(maxModeLen + 4) << unsortedMode->Description << Endl;
                } else {
                    Cerr << SeparationString << Endl;
                    Cerr << unsortedMode->Description << Endl;
                }
            }
    } else {
        for (const auto& mode : Modes) {
            if (mode.first != mode.second->Name)
                continue;  // this is an alias

            if (!mode.second->Hidden) {
                Cerr << "  " << mode.second->FormatFullName(maxModeLen + 4) << mode.second->Description << Endl;
            }
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

TVersionHandlerPtr TModChooser::GetVersionHandler() const {
    return VersionHandler;
}

bool TModChooser::IsSvnRevisionOptionDisabled() const {
    return SvnRevisionOptionDisabled;
}

int TMainClassArgs::Run(int argc, const char** argv) {
    return DoRun(NLastGetopt::TOptsParseResult(&GetOptions(), argc, argv));
}

const NLastGetopt::TOpts& TMainClassArgs::GetOptions() {
    if (Opts_.Empty()) {
        Opts_ = NLastGetopt::TOpts();
        RegisterOptions(Opts_.GetRef());
    }

    return Opts_.GetRef();
}

void TMainClassArgs::RegisterOptions(NLastGetopt::TOpts& opts) {
    opts.AddHelpOption('h');
}

int TMainClassArgs::operator()(const int argc, const char** argv) {
    return Run(argc, argv);
}

int TMainClassModes::operator()(const int argc, const char** argv) {
    return Run(argc, argv);
}

int TMainClassModes::Run(int argc, const char** argv) {
    auto& chooser = GetSubModes();
    return chooser.Run(argc, argv);
}

const TModChooser& TMainClassModes::GetSubModes() {
    if (Modes_.Empty()) {
        Modes_.ConstructInPlace();
        RegisterModes(Modes_.GetRef());
    }

    return Modes_.GetRef();
}

void TMainClassModes::RegisterModes(TModChooser& modes) {
    modes.SetModesHelpOption("-h");
}
