#include "completer_command.h"

#include "completion_generator.h"
#include "last_getopt.h"
#include "wrap.h"

#include <library/cpp/colorizer/colors.h>

#include <util/folder/dirut.h>
#include <util/folder/path.h>

#include <util/stream/file.h>
#include <util/stream/str.h>

#include <util/string/subst.h>

#include <util/system/env.h>

namespace NLastGetopt {
    namespace {

        enum class EShell {
            Bash,
            Zsh,
        };

        std::optional<EShell> ParseShell(TStringBuf value) {
            if (value == "bash") {
                return EShell::Bash;
            }
            if (value == "zsh") {
                return EShell::Zsh;
            }
            return std::nullopt;
        }

    } // namespace

    TString MakeInfo(TStringBuf command, TStringBuf flag) {
        TString info = (
            "This command generates shell script with completion function and prints it to `stdout`, "
            "allowing one to re-direct the output to the file of their choosing. "
            "Where you place the file will depend on which shell and operating system you are using."
            "\n"
            "\n"
            "\n"
            "{B}BASH (Linux){R}:"
            "\n"
            "\n"
            "For system-wide commands, completion files are stored in `/etc/bash_completion.d/`. "
            "For user commands, they are stored in `~/.local/share/bash-completion/completions`. "
            "So, pipe output of this script to a file in one of these directories:"
            "\n"
            "\n"
            "    $ mkdir -p ~/.local/share/bash-completion/completions"
            "\n"
            "    $ {command} {completion} bash >"
            "\n"
            "          ~/.local/share/bash-completion/completions/{command}"
            "\n"
            "\n"
            "You'll need to restart your shell for changes to take effect."
            "\n"
            "\n"
            "\n"
            "{B}BASH (OSX){R}:"
            "\n"
            "\n"
            "You'll need `bash-completion` (or `bash-completion@2` if you're using non-default, newer bash) "
            "homebrew formula. Completion files are stored in `/usr/local/etc/bash_completion.d`:"
            "\n"
            "\n"
            "    $ mkdir -p $(brew --prefix)/etc/bash_completion.d"
            "\n"
            "    $ {command} {completion} bash >"
            "\n"
            "          $(brew --prefix)/etc/bash_completion.d/{command}"
            "\n"
            "\n"
            "Alternatively, just source the script in your `~/.bash_profile`."
            "\n"
            "\n"
            "You'll need to restart your shell for changes to take effect."
            "\n"
            "\n"
            "\n"
            "{B}ZSH{R}:"
            "\n"
            "\n"
            "Zsh looks for completion scripts in any directory listed in `$fpath` variable. We recommend placing "
            "completions to `~/.zfunc`:"
            "\n"
            "\n"
            "    $ mkdir -m755 -p ~/.zfunc"
            "\n"
            "    $ {command} {completion} zsh > ~/.zfunc/_{command}"
            "\n"
            "\n"
            "Add the following lines to your `.zshrc` just before `compinit`:"
            "\n"
            "\n"
            "    fpath+=~/.zfunc"
            "\n"
            "\n"
            "You'll need to restart your shell for changes to take effect.");
        SubstGlobal(info, "{command}", command);
        SubstGlobal(info, "{completion}", flag);
        SubstGlobal(info, "{B}", NColorizer::StdErr().LightDefault());
        SubstGlobal(info, "{R}", NColorizer::StdErr().Reset());
        return info;
    }

    NComp::ICompleterPtr ShellChoiceCompleter() {
        return NComp::Choice({{"zsh"}, {"bash"}});
    }

    TString MakeModeInfo(const TCompletionConfig& config) {
        TStringBuilder info;
        info << "Generate shell completion for `" << config.Command << "`";
        for (const auto& alias : config.CommandAliases) {
            info << ", `" << alias << "`";
        }
        if (!config.YaToolName.empty()) {
            info << ", and `ya tool " << config.YaToolName << "`";
        }
        if (config.EnableInstaller) {
            info << ". Without `--install`, print the script to stdout.";
            info << " With `--install`, write the standalone completion";
            if (!config.YaToolName.empty()) {
                info << " and the `ya tool` completion shard";
            }
            info << " to the user data directory.";
            if (!config.YaToolName.empty()) {
                info << " Run `ya completion --install --bash` or `ya completion --install --zsh` once to enable "
                        "completion for the `ya` command itself.";
            }
        } else {
            info << ". Print the script to stdout.";
        }
        return info;
    }

    TString GenerateCompletion(
        const TModChooser* modChooser,
        const TCompletionConfig& config,
        EShell shell)
    {
        TStringStream output;
        switch (shell) {
            case EShell::Bash:
                TBashCompletionGenerator(modChooser).Generate(config, output);
                break;
            case EShell::Zsh:
                TZshCompletionGenerator(modChooser).Generate(config, output);
                break;
        }
        return std::move(output).Str();
    }

    TVector<TFsPath> GetCompletionPaths(const TCompletionConfig& config, EShell shell) {
        auto dataHome = GetEnv("XDG_DATA_HOME");
        if (dataHome.empty()) {
            dataHome = (TFsPath(GetHomeDir()) / ".local" / "share").GetPath();
        }

        TFsPath directory;
        TString fileName;
        switch (shell) {
            case EShell::Bash:
                directory = TFsPath(dataHome) / "bash-completion" / "completions";
                fileName = config.Command;
                break;
            case EShell::Zsh:
                directory = TFsPath(dataHome) / "zsh" / "site-functions";
                fileName = "_" + config.Command;
                break;
        }

        TVector<TFsPath> paths = {directory / fileName};
        if (!config.YaToolName.empty()) {
            paths.push_back(directory / "ya-tool.d" / config.YaToolName);
        }
        return paths;
    }

    void WriteCompletion(const TFsPath& path, TStringBuf completion) {
        path.Parent().MkDirs(MODE0755);
        TFileOutput output(path.GetPath());
        output.Write(completion.data(), completion.size());
    }

    TOpt MakeCompletionOpt(const TOpts* opts, TString command, TString name) {
        return TOpt()
            .AddLongName(name)
            .Help("generate tab completion script for zsh or bash")
            .CompletionHelp("generate tab completion script")
            .OptionalArgument("shell-syntax")
            .CompletionArgHelp("shell syntax for completion script")
            .IfPresentDisableCompletion()
            .Completer(ShellChoiceCompleter())
            .Handler1T<TString>([opts, command, name](TStringBuf shell) {
                if (shell.empty()) {
                    Cerr << Wrap(80, MakeInfo(command, "--" + name)) << Endl;
                } else if (shell == "bash") {
                    TBashCompletionGenerator(opts).Generate(command, Cout);
                } else if (shell == "zsh") {
                    TZshCompletionGenerator(opts).Generate(command, Cout);
                } else {
                    Cerr << "Unknown shell name " << TString{shell}.Quote() << Endl;
                    exit(1);
                }
                exit(0);
            });
    }

    class TCompleterMode: public TMainClassArgs {
    public:
        TCompleterMode(const TModChooser* modChooser, TCompletionConfig config)
            : Config_(std::move(config))
            , Modes_(modChooser)
        {
        }

    protected:
        void RegisterOptions(NLastGetopt::TOpts& opts) override {
            TMainClassArgs::RegisterOptions(opts);
            if (Config_.EnableUserFriendlyUsage) {
                opts.EnableUserFriendlyUsage();
                if (Modes_->IsSvnRevisionOptionDisabled()) {
                    if (auto* svnRevisionOption = opts.FindLongOption("svnrevision")) {
                        svnRevisionOption->Hidden();
                    }
                }
            }

            opts.SetTitle("Generate tab completion scripts for zsh or bash");

            if (Config_.EnableInstaller) {
                auto installHelp = TString("Install completion for direct invocations");
                if (!Config_.YaToolName.empty()) {
                    installHelp += " and ya tool " + Config_.YaToolName;
                }
                opts.AddLongOption("install", installHelp)
                    .NoArgument()
                    .SetFlag(&Install_);
            }

            if (Config_.EnableUserFriendlyUsage) {
                opts.AddSection("Description", MakeModeInfo(Config_));
            } else {
                opts.AddSection("Description", MakeInfo(Config_.Command, Config_.ModName));
            }

            if (Config_.EnableInstaller) {
                TStringBuilder examples;
                examples << Config_.Command << " " << Config_.ModName << " bash --install";
                if (!Config_.YaToolName.empty()) {
                    examples << "\nya tool " << Config_.YaToolName << " " << Config_.ModName << " zsh --install";
                }
                opts.SetExamples(examples);
            }

            opts.SetFreeArgsNum(1);
            auto& shellArg = opts.GetFreeArgSpec(0);
            if (Config_.EnableUserFriendlyUsage) {
                shellArg
                    .Title("SHELL")
                    .Help("Shell whose completion script should be generated: bash or zsh")
                    .CompletionArgHelp("shell");
            } else {
                shellArg
                    .Title("<shell-syntax>")
                    .Help("shell syntax  for completion script (bash or zsh)")
                    .CompletionArgHelp("shell syntax for completion script");
            }
            shellArg.Completer(ShellChoiceCompleter());
        }

        int DoRun(NLastGetopt::TOptsParseResult&& parsedOptions) override {
            auto arg = parsedOptions.GetFreeArgs()[0];
            arg.to_lower();

            const auto shell = ParseShell(arg);
            if (!shell) {
                Cerr << "Unknown shell name " << arg.Quote() << Endl;
                parsedOptions.PrintUsage();
                return 1;
            }

            auto completion = GenerateCompletion(Modes_, Config_, *shell);
            if (!Install_) {
                Cout << completion;
                return 0;
            }

            auto paths = GetCompletionPaths(Config_, *shell);
            for (const auto& path : paths) {
                WriteCompletion(path, completion);
            }
            Cout << "Installed " << arg << " completion:" << Endl;
            for (const auto& path : paths) {
                Cout << "  " << path.GetPath() << Endl;
            }
            Cout << "Restart the shell to load the updated completion." << Endl;
            return 0;
        }

    private:
        TCompletionConfig Config_;
        const TModChooser* Modes_;
        bool Install_ = false;
    };

    THolder<TMainClassArgs> MakeCompletionMod(const TModChooser* modChooser, TString command, TString modName) {
        return MakeCompletionMod(
            modChooser,
            TCompletionConfig{
                .ModName = std::move(modName),
                .Command = std::move(command),
            });
    }

    THolder<TMainClassArgs> MakeCompletionMod(const TModChooser* modChooser, TCompletionConfig config) {
        return MakeHolder<TCompleterMode>(modChooser, std::move(config));
    }
}
