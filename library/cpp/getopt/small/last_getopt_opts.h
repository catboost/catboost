#pragma once

#include "last_getopt_opt.h"

#include <library/cpp/colorizer/fwd.h>

#include <util/generic/map.h>

namespace NLastGetopt {
    enum EArgPermutation {
        REQUIRE_ORDER,
        PERMUTE,
        RETURN_IN_ORDER,
        DEFAULT_ARG_PERMUTATION = PERMUTE
    };

    /**
     * NLastGetopt::TOpts is a storage of program options' parse rules.
     * It contains information about all options, free args, some parsing options
     *    and rules about interaction between options.
     *
     * The main point for defining program options.
     *
     * The parsing rules determined by the following parts:
     *   - Arguments permutation. It is expected free args be after named args.
     *     This point adjusts how to treat breaking this expectation.
     *       if REQUIRE_ORDER is choosen, the exception during parsing will be raised,
     *            the special string " -- " will be treated as end of named
     *            options: all options after it will be parsed as free args
     *       if PERMUTE is choosen, arguments will be rearranged in correct order,
     *       if RETURN_IN_ORDER is choosen, all free args will be ommited (TODO: looks very strange)
     *   - Using '+' as a prefix instead '--' for long names
     *   - Using "-" as a prefix for both short and long names
     *   - Allowing unknown options
     *
     */
    class TOpts {
        friend class TOptsParseResult;
        friend class TOptsParser;

    public:
        static constexpr const ui32 UNLIMITED_ARGS = Max<ui32>();

        typedef TVector<TSimpleSharedPtr<TOpt>> TOptsVector;
        TOptsVector Opts_; // infomation about named (short and long) options
        TVector<std::function<void(TStringBuf)>> ArgBindings_;

        EArgPermutation ArgPermutation_ = DEFAULT_ARG_PERMUTATION; // determines how to parse positions of named and free options. See information below.
        bool AllowSingleDashForLong_ = false;                      //
        bool AllowPlusForLong_ = false;                            // using '+' instead '--' for long options

        //Allows unknwon options:
        bool AllowUnknownCharOptions_ = false;
        bool AllowUnknownLongOptions_ = false;

        ui32 Wrap_ = 80;
        bool CheckUserTypos_ = false;

    private:
        ui32 FreeArgsMin_; // minimal number of free args
        ui32 FreeArgsMax_; // maximal number of free args

        TMap<ui32, TFreeArgSpec> FreeArgSpecs_; // mapping [free arg position] -> [free arg specification]
        TFreeArgSpec TrailingArgSpec_;          // spec for the trailing argument (when arguments are unlimited)
        TString DefaultFreeArgTitle_ = "ARG"; // title that's used for free args without a title

        TString Title;              // title of the help string
        TString CustomCmdLineDescr; // user defined help string
        TString CustomUsage;        // user defined usage string

        TVector<std::pair<TString, TString>> Sections;  // additional help entries to print after usage

    public:
        /**
         * Constructs TOpts from string as in getopt(3)
         */
        TOpts(const TStringBuf& optstring = TStringBuf());

        /**
         * Constructs TOpts from string as in getopt(3) and
         * additionally adds help option (for '?') and svn-verstion option (for 'V')
         */
        static TOpts Default(const TStringBuf& optstring = TStringBuf()) {
            TOpts opts(optstring);
            opts.AddHelpOption();
            opts.AddVersionOption();
            return opts;
        }

        /**
         * Checks correctness of options' descriptions.
         * Throws TConfException if validation failed.
         * Check consist of:
         *    -not intersecting of names
         *    -compability of settings, that responsable for freeArgs parsing
         */
        void Validate() const;

        /**
         * Search for the option with given long name
         * @param name     long name for search
         * @return         ptr on result (nullptr if not found)
         */
        const TOpt* FindLongOption(const TStringBuf& name) const;

        /**
         * Search for the option with given short name
         * @param c        short name for search
         * @return         ptr on result (nullptr if not found)
         */
        const TOpt* FindCharOption(char c) const;

        /**
         * Search for the option with given long name
         * @param name     long name for search
         * @return         ptr on result (nullptr if not found)
         */
        TOpt* FindLongOption(const TStringBuf& name);

        /**
         * Search for the option with given short name
         * @param c        short name for search
         * @return         ptr on result (nullptr if not found)
         */
        TOpt* FindCharOption(char c);

        /**
         * Search for the option with given name
         * @param name     name for search
         * @return         ptr on result (nullptr if not found)
         */
        /// @{

        const TOpt* FindOption(const TStringBuf& name) const {
            return FindLongOption(name);
        }

        TOpt* FindOption(const TStringBuf& name) {
            return FindLongOption(name);
        }

        const TOpt* FindOption(char c) const {
            return FindCharOption(c);
        }

        TOpt* FindOption(char c) {
            return FindCharOption(c);
        }

        /// @}

        /**
         * Sets title of the help string
         * @param title        title to set
         */
        void SetTitle(const TString& title) {
            Title = title;
        }

        /**
         * @return true if there is an option with given long name
         *
         * @param name        long name for search
         */
        bool HasLongOption(const TString& name) const {
            return FindLongOption(name) != nullptr;
        }

        /**
         * @return true if there is an option with given short name
         *
         * @param char        short name for search
         */
        bool HasCharOption(char c) const {
            return FindCharOption(c) != nullptr;
        }

        /**
         * Search for the option with given long name
         * @param name     long name for search
         * @return         ref on result (throw exception if not found)
         */
        const TOpt& GetLongOption(const TStringBuf& name) const;

        /**
         * Search for the option with given long name
         * @param name     long name for search
         * @return         ref on result (throw exception if not found)
         */
        TOpt& GetLongOption(const TStringBuf& name);

        /**
         * Search for the option with given short name
         * @param c        short name for search
         * @return         ref on result (throw exception if not found)
         */
        const TOpt& GetCharOption(char c) const;

        /**
         * Search for the option with given short name
         * @param c        short name for search
         * @return         ref on result (throw exception if not found)
         */
        TOpt& GetCharOption(char c);

        /**
         * Search for the option with given name
         * @param name     name for search
         * @return         ref on result (throw exception if not found)
         */
        /// @{

        const TOpt& GetOption(const TStringBuf& name) const {
            return GetLongOption(name);
        }

        TOpt& GetOption(const TStringBuf& name) {
            return GetLongOption(name);
        }

        const TOpt& GetOption(char c) const {
            return GetCharOption(c);
        }

        TOpt& GetOption(char c) {
            return GetCharOption(c);
        }

        /// @}

        /**
         * @return true if short options exist
         */
        bool HasAnyShortOption() const;

        /**
         * @return true if long options exist
         */
        bool HasAnyLongOption() const;

        /**
         * Creates new [option description (TOpt)] as a copy of given one
         * @param option   source
         * @return         reference for created option
         */
        TOpt& AddOption(const TOpt& option);

        /**
         * Creates new free argument handling
         * @param name   name of free arg to show in help
         * @param target variable address to store parsing result into
         * @param help   help string to show in help
         */
        template <typename T>
        void AddFreeArgBinding(const TString& name, T& target, const TString& help = "") {
            ArgBindings_.emplace_back([&target](TStringBuf value) {
                target = FromString<T>(value);
            });

            FreeArgsMax_ = Max<ui32>(FreeArgsMax_, ArgBindings_.size());
            SetFreeArgTitle(ArgBindings_.size() - 1, name, help);
        }

        /**
         * Creates options list from string as in getopt(3)
         *
         * @param optstring   source
         */
        void AddCharOptions(const TStringBuf& optstring);

        /**
         * Creates new [option description (TOpt)] with given short name and given help string
         *
         * @param c        short name
         * @param help     help string
         * @return         reference for created option
         */
        TOpt& AddCharOption(char c, const TString& help = "") {
            return AddCharOption(c, DEFAULT_HAS_ARG, help);
        }

        /**
         * Creates new [option description (TOpt)] with given short name and given help string
         *
         * @param c        short name
         * @param help     help string
         * @return         reference for created option
         */
        TOpt& AddCharOption(char c, EHasArg hasArg, const TString& help = "") {
            TOpt option;
            option.AddShortName(c);
            option.Help(help);
            option.HasArg(hasArg);
            return AddOption(option);
        }

        /**
         * Creates new [option description (TOpt)] with given long name and given help string
         *
         * @param name     long name
         * @param help     help string
         * @return         reference for created option
         */
        TOpt& AddLongOption(const TString& name, const TString& help = "") {
            return AddLongOption(0, name, help);
        }

        /**
         * Creates new [option description (TOpt)] with given long and short names and given help string
         *
         * @param c        short name
         * @param name     long name
         * @param help     help string
         * @return         reference for created option
         */
        TOpt& AddLongOption(char c, const TString& name, const TString& help = "") {
            TOpt option;
            if (c != 0)
                option.AddShortName(c);
            option.AddLongName(name);
            option.Help(help);
            return AddOption(option);
        }

        /**
         * Creates new [option description (TOpt)] for help printing,
         *   adds appropriate handler for it
         * If "help" option already exist, will add given short name to it.
         *
         * @param c        new short name for help option
         */
        TOpt& AddHelpOption(char c = '?') {
            if (TOpt* o = FindLongOption("help")) {
                if (!o->CharIs(c))
                    o->AddShortName(c);
                return *o;
            }
            return AddLongOption(c, "help", "print usage")
                .HasArg(NO_ARGUMENT)
                .IfPresentDisableCompletion()
                .Handler(&PrintUsageAndExit);
        }

        /**
         * Set check user typos or not
         * @param check   bool flag for chosing
         */
        void SetCheckUserTypos(bool check = true) {
            CheckUserTypos_ = check;
        }

        /**
         * Creates new [option description (TOpt)] for svn-revision printing,
         *   adds appropriate handler for it.
         * If "svnversion" option already exist, will add given short name to it.
         *
         * @param c        new short name for "svnversion" option
         */
        TOpt& AddVersionOption(char c = 'V') {
            if (TOpt* o = FindLongOption("svnrevision")) {
                if (!o->CharIs(c))
                    o->AddShortName(c);
                return *o;
            }
            return AddLongOption(c, "svnrevision", "print svn version")
                .HasArg(NO_ARGUMENT)
                .IfPresentDisableCompletion()
                .Handler(&PrintVersionAndExit);
        }

        /**
         * Creates new option for generating completion shell scripts.
         *
         * @param command name of command that should be completed (typically corresponds to the executable name).
         */
        TOpt& AddCompletionOption(TString command, TString longName = "completion");

        /**
         * Creates or finds option with given short name
         *
         * @param c        new short name for search/create
         */
        TOpt& CharOption(char c) {
            const TOpt* opt = FindCharOption(c);
            if (opt != nullptr) {
                return const_cast<TOpt&>(*opt);
            } else {
                AddCharOption(c);
                return const_cast<TOpt&>(GetCharOption(c));
            }
        }

        /**
         * Indicate that some options can't appear together.
         *
         * Note: this is not transitive.
         *
         * Note: don't use this on options with default values. If option with default value wasn't specified,
         * parser will run handlers for default value, thus triggering a false-positive exclusivity check.
         */
        template <typename T1, typename T2>
        void MutuallyExclusive(T1&& opt1, T2&& opt2) {
            MutuallyExclusiveOpt(GetOption(std::forward<T1>(opt1)), GetOption(std::forward<T2>(opt2)));
        }

        /**
         * Like `MutuallyExclusive`, but accepts `TOpt`s instead of option names.
         */
        void MutuallyExclusiveOpt(TOpt& opt1, TOpt& opt2);

        /**
         * @return index of option
         *
         * @param opt        pointer of option to search
         */
        size_t IndexOf(const TOpt* opt) const;

        /**
         * Replace help string with given
         *
         * @param decr        new help string
         */
        void SetCmdLineDescr(const TString& descr) {
            CustomCmdLineDescr = descr;
        }

        /**
         * Replace usage string with given
         *
         * @param usage        new usage string
         */
        void SetCustomUsage(const TString& usage) {
            CustomUsage = usage;
        }

        /**
         * Add a section to print after the main usage spec.
         */
        void AddSection(TString title, TString text) {
            Sections.emplace_back(std::move(title), std::move(text));
        }

        /**
         * Add section with examples.
         *
         * @param examples text of this section
         */
        void SetExamples(TString examples) {
            AddSection("Examples", std::move(examples));
        }

        /**
         * Set minimal number of free args
         *
         * @param min        new value
         */
        void SetFreeArgsMin(size_t min) {
            FreeArgsMin_ = ui32(min);
        }


        /**
         * Get current minimal number of free args
         */
        ui32 GetFreeArgsMin() const {
            return FreeArgsMin_;
        }

        /**
         * Set maximal number of free args
         *
         * @param max        new value
         */
        void SetFreeArgsMax(size_t max) {
            FreeArgsMax_ = ui32(max);
            FreeArgsMax_ = Max<ui32>(FreeArgsMax_, ArgBindings_.size());
        }

        /**
         * Get current maximal number of free args
         */
        ui32 GetFreeArgsMax() const {
            return FreeArgsMax_;
        }

        /**
         * Get mapping for free args
         */
        const TMap<ui32, TFreeArgSpec>& GetFreeArgSpecs() const {
            return FreeArgSpecs_;
        }

        /**
         * Set exact expected number of free args
         *
         * @param count        new value
         */
        void SetFreeArgsNum(size_t count) {
            FreeArgsMin_ = ui32(count);
            FreeArgsMax_ = ui32(count);
        }

        /**
         * Set minimal and maximal number of free args
         *
         * @param min        new value for minimal
         * @param max        new value for maximal
         */
        void SetFreeArgsNum(size_t min, size_t max) {
            FreeArgsMin_ = ui32(min);
            FreeArgsMax_ = ui32(max);
        }

        /**
         * Set title and help string of free argument
         *
         * @param pos          index of argument
         * @param title        new value for argument title
         * @param help         new value for help string
         * @param optional     indicates that the flag's help string should be rendered as for optional flag;
         *                     does not affect actual flags parsing
         */
        void SetFreeArgTitle(size_t pos, const TString& title, const TString& help = TString(), bool optional = false);

        /**
         * Get free argument's spec for further modification.
         */
        TFreeArgSpec& GetFreeArgSpec(size_t pos);

        /**
         * Legacy, don't use. Same as `SetTrailingArgTitle`.
         * Older versions of lastgetopt didn't have destinction between default title and title
         * for the trailing argument.
         */
        void SetFreeArgDefaultTitle(const TString& title, const TString& help = TString()) {
            SetTrailingArgTitle(title, help);
        }

        /**
         * Set default title that will be used for all arguments that have no title.
         */
        void SetDefaultFreeArgTitle(TString title) {
            DefaultFreeArgTitle_ = std::move(title);
        }

        /**
         * Set default title that will be used for all arguments that have no title.
         */
        const TString& GetDefaultFreeArgTitle() const {
            return DefaultFreeArgTitle_;
        }

        /**
         * Set title and help for the trailing argument.
         *
         * This title and help are used to render the last repeated argument when max number of arguments is unlimited.
         */
        /// @{
        void SetTrailingArgTitle(TString title) {
            TrailingArgSpec_.Title(std::move(title));
        }
        void SetTrailingArgTitle(TString title, TString help) {
            TrailingArgSpec_.Title(std::move(title));
            TrailingArgSpec_.Help(std::move(help));
        }
        /// @}

        /**
         * Get spec for the trailing argument.
         *
         * This spec is used to render the last repeated argument when max number of arguments is unlimited.
         */
        /// @{
        TFreeArgSpec& GetTrailingArgSpec() {
            return TrailingArgSpec_;
        }
        const TFreeArgSpec& GetTrailingArgSpec() const {
            return TrailingArgSpec_;
        }
        /// @}

        /**
         * Set the rule of parsing single dash as prefix of long names
         *
         * @param value     new value of the option
         */
        void SetAllowSingleDashForLong(bool value) {
            AllowSingleDashForLong_ = value;
        }

        /**
         * Wrap help text at this number of characters. 0 to disable wrapping.
         */
        void SetWrap(ui32 wrap = 80) {
            Wrap_ = wrap;
        }

        /**
         * Print usage string
         *
         * @param program      prefix of result (path to the program)
         * @param os           destination stream
         * @param colors       colorizer
         */
        void PrintUsage(const TStringBuf& program, IOutputStream& os, const NColorizer::TColors& colors) const;

        /**
         * Print usage string
         *
         * @param program      prefix of result (path to the program)
         * @param os           destination stream
         */
        void PrintUsage(const TStringBuf& program, IOutputStream& os = Cout) const;

        /**
         * Get list of options in order of definition.
         */
        TVector<const TOpt*> GetOpts() const {
            auto ret = TVector<const TOpt*>(Reserve(Opts_.size()));
            for (auto& opt : Opts_) {
                ret.push_back(opt.Get());
            }
            return ret;
        }

    private:
        /**
         * @return argument title of a free argument
         *
         * @param pos     position of the argument
         */
        TStringBuf GetFreeArgTitle(size_t pos) const;

        /**
         * Print usage helper
         *
         * @param program    prefix of result (path to the program)
         * @param os         destination stream
         * @param colors     colorizer
         */
        void PrintCmdLine(const TStringBuf& program, IOutputStream& os, const NColorizer::TColors& colors) const;

        /**
         * Print usage helper
         *
         * @param os         destination stream
         * @param colors     colorizer
         */
        void PrintFreeArgsDesc(IOutputStream& os, const NColorizer::TColors& colors) const;
    };

}
