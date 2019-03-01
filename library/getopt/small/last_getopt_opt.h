#pragma once

#include "last_getopt_handlers.h"

#include <util/string/split.h>
#include <util/generic/ptr.h>
#include <util/generic/string.h>
#include <util/generic/maybe.h>
#include <util/generic/vector.h>

#include <stdarg.h>

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
        bool Hidden_ = false; // is visible in help
        TString ArgTitle_;    // the name of argument in help output
        TString Help_;        // the help string

        EHasArg HasArg_ = DEFAULT_HAS_ARG; // the argument parsing politics
        bool Required_ = false;            // option existence politics

    private:
        //Handlers information
        const void* UserValue_ = nullptr;
        TdOptVal OptionalValue_;
        TdOptVal DefaultValue_;
        TOptHandlers Handlers_;

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
         *  sets Help string into given
         *  @param help      new help string for the option
         *  @return self
         */
        TOpt& Help(const TString& help) {
            Help_ = help;
            return *this;
        }

        /**
         *  @return help string
         */
        TString GetHelp() const {
            return Help_;
        }

        /**
         *  run all handlers
         *  @param parser
         */
        void FireHandlers(const TOptsParser* parser) const;

        //TODO: handlers information
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

        template <typename TpVal, typename T, typename TpDef>
        TOpt& StoreResultT(T* target, const TpDef& def) {
            return Handler1T<TpVal>(def, NPrivate::TStoreResultFunctor<T, TpVal>(target));
        }

        template <typename T, typename TpDef>
        TOpt& StoreResult(T* target, const TpDef& def) {
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
        template <typename T>
        TOpt& AppendTo(TVector<T>* target) {
            void (TVector<T>::*functionPointer)(const T&) = &TVector<T>::push_back;
            return Handler1T<T>(std::bind(functionPointer, target, std::placeholders::_1));
        }

        // Emplaces TString arg to *target for each argument
        template <typename T>
        TOpt& EmplaceTo(TVector<T>* target) {
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
    };

    /**
     * NLastGetopt::TFreeArgSpec is a storage of data about free argument.
     * The data is help information and (maybe) linked named argument.
     *
     * The help information consists of following:
     *   help string
     *   argument name (title)
     *
     * [unimplemented] TODO: If the argument have a named alias, the fetched value will be parsed as
     *    an argument of that TOpt.
     */
    struct TFreeArgSpec {
        TFreeArgSpec() = default;
        TFreeArgSpec(const TString& title, const TString& help = TString(), const TOpt* namedAlias = nullptr)
            : Title(title)
            , Help(help)
            , NamedAlias(namedAlias)
        {
        }

        TString Title;
        TString Help;
        const TOpt* NamedAlias = nullptr; //unimplemented yet
    };

}
