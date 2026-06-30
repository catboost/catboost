#include "completer_command.h"

#include "completion_generator.h"
#include "last_getopt.h"
#include "wrap.h"

#include <library/cpp/colorizer/colors.h>

#include <util/string/subst.h>

namespace NLastGetopt {
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
        TCompleterMode(const TModChooser* modChooser, TString command, TString modName)
            : Command_(std::move(command))
            , Modes_(modChooser)
            , ModName_(std::move(modName))
        {
        }

    protected:
        void RegisterOptions(NLastGetopt::TOpts& opts) override {
            TMainClassArgs::RegisterOptions(opts);

            opts.SetTitle("Generate tab completion scripts for zsh or bash");

            opts.AddSection("Description", MakeInfo(Command_, ModName_));

            opts.SetFreeArgsNum(1);
            opts.GetFreeArgSpec(0)
                .Title("<shell-syntax>")
                .Help("shell syntax  for completion script (bash or zsh)")
                .CompletionArgHelp("shell syntax for completion script")
                .Completer(ShellChoiceCompleter());
        }

        int DoRun(NLastGetopt::TOptsParseResult&& parsedOptions) override {
            auto arg = parsedOptions.GetFreeArgs()[0];
            arg.to_lower();

            if (arg == "bash") {
                TBashCompletionGenerator(Modes_).Generate(Command_, Cout);
            } else if (arg == "zsh") {
                TZshCompletionGenerator(Modes_).Generate(Command_, Cout);
            } else {
                Cerr << "Unknown shell name " << arg.Quote() << Endl;
                parsedOptions.PrintUsage();
                return 1;
            }

            return 0;
        }

    private:
        TString Command_;
        const TModChooser* Modes_;
        TString ModName_;
    };

    THolder<TMainClassArgs> MakeCompletionMod(const TModChooser* modChooser, TString command, TString modName) {
        return MakeHolder<TCompleterMode>(modChooser, std::move(command), std::move(modName));
    }
}
