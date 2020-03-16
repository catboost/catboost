#pragma once

#include "formatted_output.h"

#include <util/generic/strbuf.h>
#include <util/generic/hash.h>

#include <utility>

namespace NLastGetopt::NComp {
    class ICompleter;

    class TCompleterManager {
    public:
        TCompleterManager(TStringBuf command);

        /// Register new completer and get its function name.
        TStringBuf GetCompleterID(const ICompleter* completer);

        /// Generate zsh code for all collected completers.
        void GenerateZsh(TFormattedOutput& out);

    private:
        TStringBuf Command_;
        size_t Id_;
        TVector<std::pair<TString, const ICompleter*>> Queue_;
    };

    class ICompleter {
    public:
        virtual ~ICompleter() = default;

    public:
        /// Generate arbitrary bash code that modifies `COMPREPLY`.
        virtual void GenerateBash(TFormattedOutput& out) const = 0;

        /// Generate action that will be used with `_arguments`. If this completer requires a separate function,
        /// register it in the given manager and return function name assigned by manager.
        /// Supported forms are '()', '(items...)', '((items...))', 'command ...' and ' command ...'.
        /// Other forms, such as '{eval-string}', '->state', '=action' are not supported.
        virtual TStringBuf GenerateZshAction(TCompleterManager& manager) const = 0;

        /// Generate body of a zsh function (if Action points to a custom function).
        virtual void GenerateZsh(TFormattedOutput& out, TCompleterManager& manager) const = 0;
    };

    using ICompleterPtr = TSimpleSharedPtr<ICompleter>;

    /// Generate default completions.
    /// Output of this completer depends on shell settings.
    /// Usually ut generates file paths.
    ICompleterPtr Default();

    struct TAlternative {
        /// Description for this group of completions. Leave empty to use completer's default description.
        TString Description;

        /// Completer that generates values
        ICompleterPtr Completer;

        TAlternative(ICompleterPtr completer)
            : Description("")
            , Completer(std::move(completer))
        {
        }

        TAlternative(TString description, ICompleterPtr completer)
            : Description(std::move(description))
            , Completer(std::move(completer))
        {
        }
    };

    /// Run multiple completers and unite their output.
    /// Each completer's output placed in a separate group with its own description.
    ICompleterPtr Alternative(TVector<TAlternative> alternatives);

    struct TChoice {
        /// Option value.
        TString Choice;

        /// Description for a value.
        TString Description = "";

        TChoice(TString choice)
            : Choice(std::move(choice))
        {
        }

        TChoice(TString choice, TString description)
            : Choice(std::move(choice))
            , Description(std::move(description))
        {
        }
    };

    /// Complete items from a predefined list of choices.
    ICompleterPtr Choice(TVector<TChoice> choices);

    /// Complete files and directories. May filter results by pattern, e.g. `*.txt`.
    ICompleterPtr File(TString pattern= "");

    /// Complete directories.
    ICompleterPtr Directory();

    /// Complete hosts.
    ICompleterPtr Host();

    /// Complete process IDs.
    ICompleterPtr Pid();

    /// Complete users that're found in the system.
    ICompleterPtr User();

    /// Complete user groups that're found in the system.
    /// N: for some reason,
    ICompleterPtr Group();

    /// Complete URLs.
    ICompleterPtr Url();

    /// Complete TTY interfaces.
    ICompleterPtr Tty();

    /// Complete network interfaces.
    ICompleterPtr NetInterface();

    /// Complete timezone identifiers.
    ICompleterPtr TimeZone();

    /// Complete unix signal identifiers, e.g. `ABRT` or `KILL`.
    ICompleterPtr Signal();

    /// Complete domains.
    ICompleterPtr Domain();

    /// Custom completer. See `LaunchSelf` below.
    class TCustomCompleter {
    public:
        static void FireCustomCompleter(int argc, const char** argv);
        static void RegisterCustomCompleter(TCustomCompleter* completer) noexcept;

        struct TReg {
            TReg(TCustomCompleter* completer) noexcept {
                TCustomCompleter::RegisterCustomCompleter(completer);
            }
        };

    public:
        virtual ~TCustomCompleter() = default;

    public:
        virtual void GenerateCompletions(int argc, const char** argv) = 0;
        virtual TStringBuf GetUniqueName() const = 0;

    protected:
        void AddCompletion(TStringBuf completion);

    private:
        TCustomCompleter* Next_;
    };

#define Y_COMPLETER(N)                                                  \
class T##N: public ::NLastGetopt::NComp::TCustomCompleter {             \
    public:                                                             \
        void GenerateCompletions(int argc, const char** argv) override; \
        TStringBuf GetUniqueName() const override { return #N; }        \
    };                                                                  \
    T##N N = T##N();                                                    \
    ::NLastGetopt::NComp::TCustomCompleter::TReg _Reg_##N = &N;         \
    void T##N::GenerateCompletions(Y_DECLARE_UNUSED int argc, Y_DECLARE_UNUSED const char** argv)

    /// Launches this binary with a specially formed flags.
    ///
    /// Your application must be set up in a certain way for this to work.
    ///
    /// First, create a custom completer:
    ///
    /// ```
    /// Y_COMPLETER(SomeUniqueName) {
    ///     AddCompletion("foo");
    ///     AddCompletion("bar");
    ///     AddCompletion("baz");
    /// }
    /// ```
    ///
    /// Then, use it with this function.
    ///
    /// On completion attempt, completer will call your binary with some special arguments.
    ///
    /// In your main, before any other logic, call `TCustomCompleter::FireCustomCompleter`. This function will
    /// check for said special arguments and invoke the right completer:
    ///
    /// ```
    /// int main(int argc, const char** argv) {
    ///     TCustomCompleter::FireCustomCompleter(argc, argv);
    ///     ...
    /// }
    /// ```
    ICompleterPtr LaunchSelf(TCustomCompleter& completer);
}
