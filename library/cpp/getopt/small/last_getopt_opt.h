#pragma once

#include "completer.h"
#include "last_getopt_handlers.h"

#include <util/string/split.h>
#include <util/generic/hash_set.h>
#include <util/generic/ptr.h>
#include <util/generic/string.h>
#include <util/generic/maybe.h>
#include <util/generic/vector.h>
#include <util/string/cast.h>
#include <util/string/join.h>

#include <optional>

namespace NLastGetopt {
    enum EHasArg {
        NO_ARGUMENT,
        REQUIRED_ARGUMENT,
        OPTIONAL_ARGUMENT,
        DEFAULT_HAS_ARG = REQUIRED_ARGUMENT
    };

    /**
     * NLastGetopt::TOpt is a storage of data about exactly one program option.
     * The data is: parse politics and help information.
     *
     * The help information consists of following:
     *   hidden or visible in help information
     *   help string
     *   argument name
     *
     * Parse politics is determined by following parameters:
     *   argument parse politics: no/optional/required/
     *   option existence: required or optional
     *   handlers. See detailed documentation: <TODO:link>
     *   default value: if the option has argument, but the option is ommited,
     *                     then the <default value> is used as the value of the argument
     *   optional value: if the option has optional-argument, the option is present in parsed string,
     *                      but the argument is omitted, then <optional value is used>
     *      in case of "not given <optional value>, omited optional argument" the <default value> is used
     *   user value: allows to store arbitary pointer for handlers
     */
    class TOpt {
    public:
        typedef TVector<char> TShortNames;
        typedef TVector<TString> TLongNames;

    protected:
        TShortNames Chars_;
        TLongNames LongNames_;

    private:
        typedef TMaybe<TString> TdOptVal;
        typedef TVector<TSimpleSharedPtr<IOptHandler>> TOptHandlers;

    public:
        bool Hidden_ = false;       // is visible in help
        TString ArgTitle_;          // the name of argument in help output
        TString Help_;              // the help string
        TString CompletionHelp_;    // the help string that's used in completion script, a shorter version of Help_
        TString CompletionArgHelp_; // the description of argument in completion script

        EHasArg HasArg_ = DEFAULT_HAS_ARG; // the argument parsing politics
        bool Required_ = false;            // option existence politics
        bool EqParseOnly_ = false;             // allows option not to read argument

        bool AllowMultipleCompletion_ = false; // let the completer know that this option can occur more than once

        bool DisableCompletionForOptions_ = false;
        bool DisableCompletionForFreeArgs_ = false;
        TShortNames DisableCompletionForChar_;
        TLongNames DisableCompletionForLongName_;
        TVector<size_t> DisableCompletionForFreeArg_;
        NComp::ICompleterPtr Completer_;

    private:
        //Handlers information
        const void* UserValue_ = nullptr;
        TdOptVal OptionalValue_;
        TdOptVal DefaultValue_;
        TOptHandlers Handlers_;
        THashSet<TString> Choices_;

    public:
        /**
         *  Checks if given char can be a short name
         *  @param c               char to check
         */
        static bool IsAllowedShortName(unsigned char c);

        /**
         *  Checks if given string can be a long name
         *  @param name            string to check
         *  @param c               if given, the first bad charecter will be saved in c
         */
        static bool IsAllowedLongName(const TString& name, unsigned char* c = nullptr);

        /**
         *  @return one of the expected representations of the option.
         *  If the option has short names, will return "-<char>"
         *  Otherwise will return "--<long name>"
         */
        TString ToShortString() const;

        /**
         *  check if given string is one of the long names
         *
         *  @param name               string to check
         */
        bool NameIs(const TString& name) const;

        /**
         *  check if given char is one of the short names
         *
         *  @param c               char to check
         */
        bool CharIs(char c) const;

        /**
         *  If string has long names - will return one of them
         *  Otherwise will throw
         */
        TString GetName() const;

        /**
         *  adds short alias for the option
         *
         *  @param c               new short name
         *
         *  @return self
         */
        TOpt& AddShortName(unsigned char c);

        /**
         *  return all short names of the option
         */
        const TShortNames& GetShortNames() const {
            return Chars_;
        }

        /**
         *  adds long alias for the option
         *
         *  @param name              new long name
         *
         *  @return self
         */
        TOpt& AddLongName(const TString& name);

        /**
         *  return all long names of the option
         */
        const TLongNames& GetLongNames() const {
            return LongNames_;
        }

        /**
         *  @return one of short names of the opt. If there is no short names exception is raised.
         */
        char GetChar() const;

        /**
         *  @return one of short names of the opt. If there is no short names '\0' returned.
         */
        char GetCharOr0() const;

        /**
         *  @returns argument parsing politics
         */
        const EHasArg& GetHasArg() const {
            return HasArg_;
        }

        /**
         *  sets argument parsing politics
         *
         *  Note: its better use one of RequiredArgument/NoArgument/OptionalArgument methods
         *
         *  @param hasArg      new argument parsing mode
         *  @return self
         */
        TOpt& HasArg(EHasArg hasArg) {
            HasArg_ = hasArg;
            return *this;
        }

        /**
         *  @returns argument title
         */
        TString GetArgTitle() const {
            return ArgTitle_;
        }

        /**
         *  sets argument parsing politics into REQUIRED_ARGUMENT
         *
         *  @param title      the new name of argument in help output
         *  @return self
         */
        TOpt& RequiredArgument(const TString& title = "") {
            ArgTitle_ = title;
            return HasArg(REQUIRED_ARGUMENT);
        }

        /**
         *  sets argument parsing politics into NO_ARGUMENT
         *
         *  @return self
         */
        TOpt& NoArgument() {
            return HasArg(NO_ARGUMENT);
        }

        /**
         *  sets argument parsing politics into OPTIONAL_ARGUMENT
         *  for details see NLastGetopt::TOpt
         *
         *  @param title      the new name of argument in help output
         *  @return self
         */
        TOpt& OptionalArgument(const TString& title = "") {
            ArgTitle_ = title;
            return HasArg(OPTIONAL_ARGUMENT);
        }

        /**
         *  sets argument parsing politics into OPTIONAL_ARGUMENT
         *  sets the <optional value> into given
         *
         *  for details see NLastGetopt::TOpt
         *
         *  @param val        the new <optional value>
         *  @param title      the new name of argument in help output
         *  @return self
         */
        TOpt& OptionalValue(const TString& val, const TString& title = "") {
            OptionalValue_ = val;
            return OptionalArgument(title);
        }

        /**
         *  checks if "argument parsing politics" is OPTIONAL_ARGUMENT and the <optional value> is set.
         */
        bool HasOptionalValue() const {
            return OPTIONAL_ARGUMENT == HasArg_ && OptionalValue_;
        }

        /**
         *  @return optional value
         *  throws exception if optional value wasn't set
         */
        const TString& GetOptionalValue() const {
            return *OptionalValue_;
        }

        /**
         *  sets <default value>
         *  @return self
         */
        template <typename T>
        TOpt& DefaultValue(const T& val) {
            DefaultValue_ = ToString(val);
            return *this;
        }

        /**
         *  checks if default value is set.
         */
        bool HasDefaultValue() const {
            return DefaultValue_.Defined();
        }

        /**
         *  @return default value
         *  throws exception if <default value> wasn't set
         */
        const TString& GetDefaultValue() const {
            return *DefaultValue_;
        }

        /**
         *  sets the option to be required
         *  @return self
         */
        TOpt& Required() {
            Required_ = true;
            return *this;
        }

        /**
         *  allow only --option=arg parsing and disable --option arg
         *  @return self
         */
        TOpt& DisableSpaceParse() {
            Y_ASSERT(GetHasArg() == OPTIONAL_ARGUMENT || GetHasArg() == REQUIRED_ARGUMENT);
            EqParseOnly_ = true;
            return *this;
        }

        /**
         *  @return true if only --option=arg parse allowed
         */
        bool IsEqParseOnly() const {
            return EqParseOnly_;
        }

        /**
         *  sets the option to be optional
         *  @return self
         */
        TOpt& Optional() {
            Required_ = false;
            return *this;
        }

        /**
         *  @return true if the option is required
         */
        bool IsRequired() const {
            return Required_;
        }

        /**
         *  sets the option to be hidden (invisible in help)
         *  @return self
         */
        TOpt& Hidden() {
            Hidden_ = true;
            return *this;
        }

        /**
         *  @return true if the option is hidden
         */
        bool IsHidden() const {
            return Hidden_;
        }

        /**
         *  sets the <user value>
         *  @return self
         *  for details see NLastGetopt::TOpt
         */
        TOpt& UserValue(const void* userval) {
            UserValue_ = userval;
            return *this;
        }

        /**
         *  @return user value
         */
        const void* UserValue() const {
            return UserValue_;
        }

        /**
         * Set help string that appears with `--help`. Unless `CompletionHelp` is given, this message will also be used
         * in completion script. In this case, don't make it too long, don't start it with a capital letter and don't
         * end it with a full stop.
         *
         * Note that `Help`, `CompletionHelp` and `CompletionArgHelp` are not the same. `Help` is printed in program
         * usage (when you call `program --help`), `CompletionHelp` is printed when completer lists available
         * options, and `CompletionArgHelp` is printed when completer shows available values for the option.
         *
         * Example of good help message:
         *
         * ```
         * opts.AddLongOption('t', "timeout")
         *     .Help("specify query timeout in milliseconds")
         *     .CompletionHelp("specify query timeout")
         *     .CompletionArgHelp("query timeout (ms) [default=500]");
         * ```
         *
         * Notice how `Help` and `CompletionArgHelp` have units in them, but `CompletionHelp` don't.
         *
         * Another good example is the help option:
         *
         * ```
         * opts.AddLongOption('h', "help")
         *     .Help("print this message and exit")
         *     .CompletionHelp("print help message and exit");
         * ```
         *
         * Notice how `Help` mentions 'this message', but `CompletionHelp` mentions just 'help message'.
         *
         * See more on completion descriptions codestyle:
         * https://github.com/zsh-users/zsh/blob/master/Etc/completion-style-guide#L43
         */
        TOpt& Help(const TString& help) {
            Help_ = help;
            return *this;
        }

        /**
         * Get help string.
         */
        const TString& GetHelp() const {
            return Help_;
        }

        TString GetChoicesHelp() const {
            return JoinSeq(", ", Choices_);
        }

        /**
         * Set help string that appears when argument completer lists available options.
         *
         * See `Help` function for info on how this is different from setting `Help` and `CompletionArgHelp`.
         *
         * Use shorter messages for this message. Don't start them with a capital letter and don't end them
         * with a full stop. De aware that argument name and default value will not be printed by completer.
         *
         * In zsh, these messages will look like this:
         *
         * ```
         * $ program -<tab><tab>
         *  -- option --
         * --help    -h  -- print help message and exit
         * --timeout -t  -- specify query timeout
         * ```
         */
        TOpt& CompletionHelp(const TString& help) {
            CompletionHelp_ = help;
            return *this;
        }

        /**
         * Get help string that appears when argument completer lists available options.
         */
        const TString& GetCompletionHelp() const {
            return CompletionHelp_ ? CompletionHelp_ : Help_;
        }

        /**
         * Set help string that appears when completer suggests available values.
         *
         * See `Help` function for info on how this is different from setting `Help` and `CompletionHelp`.
         *
         * In zsh, these messages will look like this:
         *
         * ```
         * $ program --timeout <tab><tab>
         *  -- query timeout (ms) [default=500] --
         * 50     100     250     500     1000
         * ```
         */
        TOpt& CompletionArgHelp(const TString& help) {
            CompletionArgHelp_ = help;
            return *this;
        }

        /**
         *  @return argument help string for use in completion script.
         */
        const TString& GetCompletionArgHelp() const {
            return CompletionArgHelp_ ? CompletionArgHelp_ : ArgTitle_;
        }

        /**
         * Let the completer know that this option can occur more than once.
         */
        TOpt& AllowMultipleCompletion(bool allowMultipleCompletion = true) {
            AllowMultipleCompletion_ = allowMultipleCompletion;
            return *this;
        }

        /**
         * @return true if completer will offer completion for this option multiple times.
         */
        bool MultipleCompletionAllowed() const {
            return AllowMultipleCompletion_;
        }

        /**
         * Tell the completer to disable further completion if this option is present.
         * This is useful for options like `--help`.
         *
         * Note: this only works in zsh.
         *
         * @return self
         */
        TOpt& IfPresentDisableCompletion(bool value = true) {
            IfPresentDisableCompletionForOptions(value);
            IfPresentDisableCompletionForFreeArgs(value);
            return *this;
        }

        /**
         * Tell the completer to disable completion for all options if this option is already present in the input.
         * Free arguments will still be completed.
         *
         * Note: this only works in zsh.
         *
         * @return self
         */
        TOpt& IfPresentDisableCompletionForOptions(bool value = true) {
            DisableCompletionForOptions_ = value;
            return *this;
        }

        /**
         * Tell the completer to disable option `c` if this option is already present in the input.
         * For example, if you have two options `-a` and `-r` that are mutually exclusive, disable `-r` for `-a` and
         * disable `-a` for `-r`, like this:
         *
         * ```
         * opts.AddLongOption('a', "acquire").IfPresentDisableCompletionFor('r');
         * opts.AddLongOption('r', "release").IfPresentDisableCompletionFor('a');
         * ```
         *
         * This way, if user enabled option `-a`, completer will not suggest option `-r`.
         *
         * Note that we don't have to disable all flags for a single option. That is, disabling `-r` in the above
         * example disables `--release` automatically.
         *
         * Note: this only works in zsh.
         *
         * @param c char option that should be disabled when completer hits this option.
         */
        TOpt& IfPresentDisableCompletionFor(char c) {
            DisableCompletionForChar_.push_back(c);
            return *this;
        }

        /**
         * Like `IfPresentDisableCompletionFor(char c)`, but for long options.
         */
        TOpt& IfPresentDisableCompletionFor(const TString& name) {
            DisableCompletionForLongName_.push_back(name);
            return *this;
        }

        /**
         * Like `IfPresentDisableCompletionFor(char c)`, but for long options.
         */
        TOpt& IfPresentDisableCompletionFor(const TOpt& opt);

        /**
         * Tell the completer to disable completion for the given free argument if this option is present.
         *
         * Note: this only works in zsh.
         *
         * @param arg index of free arg
         */
        TOpt& IfPresentDisableCompletionForFreeArg(size_t index) {
            DisableCompletionForFreeArg_.push_back(index);
            return *this;
        }

        /**
         * Assign a completer for this option.
         */
        TOpt& Completer(NComp::ICompleterPtr completer) {
            Completer_ = std::move(completer);
            return *this;
        }

        /**
         * Tell the completer to disable completion for the all free arguments if this option is present.
         *
         * Note: this only works in zsh.
         */
        TOpt& IfPresentDisableCompletionForFreeArgs(bool value = true) {
            DisableCompletionForFreeArgs_ = value;
            return *this;
        }

        /**
         * Run handlers for this option.
         */
        void FireHandlers(const TOptsParser* parser) const;

    private:
        TOpt& HandlerImpl(IOptHandler* handler) {
            Handlers_.push_back(handler);
            return *this;
        }

    public:
        template <typename TpFunc>
        TOpt& Handler0(TpFunc func) { // functor taking no parameters
            return HandlerImpl(new NPrivate::THandlerFunctor0<TpFunc>(func));
        }

        template <typename TpFunc>
        TOpt& Handler1(TpFunc func) { // functor taking one parameter
            return HandlerImpl(new NPrivate::THandlerFunctor1<TpFunc>(func));
        }
        template <typename TpArg, typename TpFunc>
        TOpt& Handler1T(TpFunc func) {
            return HandlerImpl(new NPrivate::THandlerFunctor1<TpFunc, TpArg>(func));
        }
        template <typename TpArg, typename TpFunc>
        TOpt& Handler1T(const TpArg& def, TpFunc func) {
            return HandlerImpl(new NPrivate::THandlerFunctor1<TpFunc, TpArg>(func, def));
        }
        template <typename TpArg, typename TpArg2, typename TpFunc>
        TOpt& Handler1T2(const TpArg2& def, TpFunc func) {
            return HandlerImpl(new NPrivate::THandlerFunctor1<TpFunc, TpArg>(func, def));
        }

        TOpt& Handler(void (*f)()) {
            return Handler0(f);
        }
        TOpt& Handler(void (*f)(const TOptsParser*)) {
            return Handler1(f);
        }

        TOpt& Handler(TAutoPtr<IOptHandler> handler) {
            return HandlerImpl(handler.Release());
        }

        template <typename T> // T extends IOptHandler
        TOpt& Handler(TAutoPtr<T> handler) {
            return HandlerImpl(handler.Release());
        }

        // Stores FromString<T>(arg) in *target
        // T maybe anything with FromString<T>(const TStringBuf&) defined
        template <typename TpVal, typename T>
        TOpt& StoreResultT(T* target) {
            return Handler1T<TpVal>(NPrivate::TStoreResultFunctor<T, TpVal>(target));
        }

        template <typename T>
        TOpt& StoreResult(T* target) {
            return StoreResultT<T>(target);
        }

        // Uses TMaybe<T> to store FromString<T>(arg)
        template <typename T>
        TOpt& StoreResult(TMaybe<T>* target) {
            return StoreResultT<T>(target);
        }

        template <typename T>
        TOpt& StoreResult(std::optional<T>* target) {
            return StoreResultT<T>(target);
        }

        template <typename TpVal, typename T, typename TpDef>
        TOpt& StoreResultT(T* target, const TpDef& def) {
            return Handler1T<TpVal>(def, NPrivate::TStoreResultFunctor<T, TpVal>(target));
        }

        template <typename T, typename TpDef>
        TOpt& StoreResult(T* target, const TpDef& def) {
            return StoreResultT<T>(target, def);
        }

        template <typename T>
        TOpt& StoreResultDef(T* target) {
            DefaultValue_ = ToString(*target);
            return StoreResultT<T>(target, *target);
        }

        template <typename T, typename TpDef>
        TOpt& StoreResultDef(T* target, const TpDef& def) {
            DefaultValue_ = ToString(def);
            return StoreResultT<T>(target, def);
        }

        // Sugar for storing flags (option without arguments) to boolean vars
        TOpt& SetFlag(bool* target) {
            return DefaultValue("0").StoreResult(target, true);
        }

        // Similar to store_true in Python's argparse
        TOpt& StoreTrue(bool* target) {
            return NoArgument().SetFlag(target);
        }

        // Similar to store_false in Python's argparse
        TOpt& StoreFalse(bool* target) {
            return NoArgument().StoreResult(target, false);
        }

        template <typename TpVal, typename T, typename TpFunc>
        TOpt& StoreMappedResultT(T* target, const TpFunc& func) {
            return Handler1T<TpVal>(NPrivate::TStoreMappedResultFunctor<T, TpFunc, TpVal>(target, func));
        }

        template <typename T, typename TpFunc>
        TOpt& StoreMappedResult(T* target, const TpFunc& func) {
            return StoreMappedResultT<T>(target, func);
        }

        // Stores given value in *target if the option is present.
        // TValue must be a copyable type, constructible from TParam.
        // T must be a copyable type, assignable from TValue.
        template <typename TValue, typename T, typename TParam>
        TOpt& StoreValueT(T* target, const TParam& value) {
            return Handler1(NPrivate::TStoreValueFunctor<T, TValue>(target, value));
        }

        // save value as target type
        template <typename T, typename TParam>
        TOpt& StoreValue(T* target, const TParam& value) {
            return StoreValueT<T>(target, value);
        }

        // save value as its original type (2nd template parameter)
        template <typename T, typename TValue>
        TOpt& StoreValue2(T* target, const TValue& value) {
            return StoreValueT<TValue>(target, value);
        }

        // Appends FromString<T>(arg) to *target for each argument
        template<class Container>
        TOpt& AppendTo(Container* target) {
            return Handler1T<typename Container::value_type>([target](auto&& value) { target->push_back(std::forward<decltype(value)>(value)); });
        }

        // Appends FromString<T>(arg) to *target for each argument
        template <typename T>
        TOpt& InsertTo(THashSet<T>* target) {
            return Handler1T<T>([target](auto&& value) { target->insert(std::forward<decltype(value)>(value)); });
        }

        // Appends FromString<T>(arg) to *target for each argument
        template <class Container>
        TOpt& InsertTo(Container* target) {
            return Handler1T<typename Container::value_type>([target](auto&& value) { target->insert(std::forward<decltype(value)>(value)); });
        }

        // Emplaces TString arg to *target for each argument
        template <typename T>
        TOpt& EmplaceTo(TVector<T>* target) {
            return Handler1T<TString>([target](TString arg) { target->emplace_back(std::move(arg)); } );
        }

        // Emplaces TString arg to *target for each argument
        template <class Container>
        TOpt& EmplaceTo(Container* target) {
            return Handler1T<TString>([target](TString arg) { target->emplace_back(std::move(arg)); } );
        }

        template <class Container>
        TOpt& SplitHandler(Container* target, const char delim) {
            return Handler(new NLastGetopt::TOptSplitHandler<Container>(target, delim));
        }

        template <class Container>
        TOpt& RangeSplitHandler(Container* target, const char elementsDelim, const char rangesDelim) {
            return Handler(new NLastGetopt::TOptRangeSplitHandler<Container>(target, elementsDelim, rangesDelim));
        }

        template <class TpFunc>
        TOpt& KVHandler(TpFunc func, const char kvdelim = '=') {
            return Handler(new NLastGetopt::TOptKVHandler<TpFunc>(func, kvdelim));
        }

        template <typename TIterator>
        TOpt& Choices(TIterator begin, TIterator end) {
            return Choices(THashSet<typename TIterator::value_type>{begin, end});
        }

        template <typename TValue>
        TOpt& Choices(THashSet<TValue> choices) {
            Choices_ = std::move(choices);
            return Handler1T<TValue>(
                [this] (const TValue& arg) {
                    if (!Choices_.contains(arg)) {
                        throw TUsageException() << " value '" << arg
                                                << "' is not allowed for option '" << GetName() << "'";
                    }
                });
        }

        TOpt& Choices(TVector<TString> choices) {
            return Choices(
                THashSet<TString>{
                    std::make_move_iterator(choices.begin()),
                    std::make_move_iterator(choices.end())
                });
        }

        TOpt& ChoicesWithCompletion(TVector<NComp::TChoice> choices) {
            Completer(NComp::Choice(choices));
            THashSet<TString> choicesSet;
            choicesSet.reserve(choices.size());
            for (const auto& choice : choices) {
                choicesSet.insert(choice.Choice);
            }
            return Choices(std::move(choicesSet));
        }
    };

    /**
     * NLastGetopt::TFreeArgSpec is a storage of data about free argument.
     * The data is help information and (maybe) linked named argument.
     *
     * The help information consists of following:
     *   help string
     *   argument name (title)
     */
    struct TFreeArgSpec {
        TFreeArgSpec() = default;
        TFreeArgSpec(const TString& title, const TString& help = TString(), bool optional = false)
            : Title_(title)
            , Help_(help)
            , Optional_(optional)
        {
        }

        TString Title_;
        TString Help_;
        TString CompletionArgHelp_;

        bool Optional_ = false;
        NComp::ICompleterPtr Completer_ = nullptr;

        /**
         * Check if this argument have default values for its title and help.
         */
        bool IsDefault() const {
            return Title_.empty() && Help_.empty();
        }

        /**
         * Set argument title.
         */
        TFreeArgSpec& Title(TString title) {
            Title_ = std::move(title);
            return *this;
        }

        /**
         * Get argument title. If title is empty, returns a default one.
         */
        TStringBuf GetTitle(TStringBuf defaultTitle) const {
            return Title_ ? TStringBuf(Title_) : defaultTitle;
        }

        /**
         * Set help string that appears with `--help`. Unless `CompletionHelp` is given, this message will also be used
         * in completion script. In this case, don't make it too long, don't start it with a capital letter and don't
         * end it with a full stop.
         *
         * See `TOpt::Help` function for more on how `Help` and `CompletionArgHelp` differ one from another.
         */
        TFreeArgSpec& Help(TString help) {
            Help_ = std::move(help);
            return *this;
        }

        /**
         * Get help string that appears with `--help`.
         */
        TStringBuf GetHelp() const {
            return Help_;
        }

        /**
         * Set help string that appears when completer suggests values fot this argument.
         */
        TFreeArgSpec& CompletionArgHelp(TString completionArgHelp) {
            CompletionArgHelp_ = std::move(completionArgHelp);
            return *this;
        }

        /**
         * Get help string that appears when completer suggests values fot this argument.
         */
        TStringBuf GetCompletionArgHelp(TStringBuf defaultTitle) const {
            return CompletionArgHelp_ ? TStringBuf(CompletionArgHelp_) : GetTitle(defaultTitle);
        }

        /**
         * Mark this argument as optional. This setting only affects help printing, it doesn't affect parsing.
         */
        TFreeArgSpec& Optional(bool optional = true) {
            Optional_ = optional;
            return *this;
        }

        /**
         * Check if this argument is optional.
         */
        bool IsOptional() const {
            return Optional_;
        }

        /**
         * Set completer for this argument.
         */
        TFreeArgSpec& Completer(NComp::ICompleterPtr completer) {
            Completer_ = std::move(completer);
            return *this;
        }
    };
}
