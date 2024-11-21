#pragma once

#include "last_getopt_opts.h"
#include "last_getopt_parser.h"

namespace NLastGetopt {
    /**
     * NLastGetopt::TOptParseResult contains all arguments for exactly one TOpt,
     * that have been fetched during parsing
     *
     * The class is a wraper over a vector of nil-terminated strings.
     */
    class TOptParseResult {
    public:
        typedef TVector<const char*> TValues;

    public:
        TOptParseResult(const TOpt* opt = nullptr)
            : Opt_(opt)
        {
        }

    public:
        const TOpt& Opt() const {
            return *Opt_;
        }
        const TOpt* OptPtr() const {
            return Opt_;
        }
        const TValues& Values() const {
            return Values_;
        }
        bool Empty() const {
            return Values().empty();
        }
        size_t Count() const {
            return Values_.size();
        }
        void AddValue(const char* val) {
            if (nullptr != val)
                Values_.push_back(val);
        }
        const char* DefVal(const char* def = nullptr) const {
            return Opt().HasDefaultValue() ? Opt().GetDefaultValue().c_str() : def;
        }
        const char* Front(const char* def = nullptr) const {
            return Empty() ? DefVal(def) : Values().front();
        }
        const char* Back(const char* def = nullptr) const {
            return Empty() ? DefVal(def) : Values().back();
        }

    private:
        const TOpt* Opt_;
        TValues Values_;
    };

    /**
     * NLastGetopt::TOptsParseResult contains result of parsing argc,argv be parser.
     *
     * In most common case constructed by argc,argv pair and rules (TOpts).
     * The instance being constructed validates rules and performs parsing, stores result for futher access.
     *
     * If error during parsing occures, the program aborts with exit code 1.
     * Note, that if PERMUTE mode is on, then data, pointed by argv can be changed.
     */
    class TOptsParseResult {
    private:
        THolder<TOptsParser> Parser_; //The instance of parser.

        // XXX: make argc, argv
        typedef TVector<TOptParseResult> TdVec;

        TdVec Opts_;    //Parsing result for all options, that have been explicitly defined in argc/argv
        TdVec OptsDef_; //Parsing result for options, that have been defined by default values only

    private:
        TOptParseResult& OptParseResult();

        /**
         * Searchs for object in given container
         *
         * @param vec     container
         * @param opt     ptr for required object
         *
         * @retunr        ptr on corresponding TOptParseResult
         */
        static const TOptParseResult* FindParseResult(const TdVec& vec, const TOpt* opt);

    protected:
        /**
         * Performs parsing of comand line arguments.
         */
        void Init(const TOpts* options, int argc, const char** argv);

        TOptsParseResult() = default;

    public:
        /**
         * The action in case of parser failure.
         * Allows to asjust behavior in derived classes.
         * By default prints error string and aborts the program
         */
        virtual void HandleError() const;

        /**
         * Constructs object by parsing arguments with given rules
         *
         * @param options      ptr on parsing rules
         * @param argc
         * @param argv
         */
        TOptsParseResult(const TOpts* options, int argc, const char* argv[]) {
            Init(options, argc, argv);
        }

        /**
         * Constructs object by parsing arguments with given rules
         *
         * @param options      ptr on parsing rules
         * @param argc
         * @param argv
         */
        TOptsParseResult(const TOpts* options, int argc, char* argv[]) {
            Init(options, argc, const_cast<const char**>(argv));
        }

        virtual ~TOptsParseResult() = default;

        /**
         * Search for TOptParseResult that corresponds to given option (TOpt)
         *
         * @param opt                ptr on required object
         * @param includeDefault     search in results obtained from default values
         *
         * @return                   ptr on result
         */
        const TOptParseResult* FindOptParseResult(const TOpt* opt, bool includeDefault = false) const;

        /**
         * Search for TOptParseResult that corresponds to given long name
         *
         * @param name               long name of required object
         * @param includeDefault     search in results obtained from default values
         *
         * @return                   ptr on result
         */
        const TOptParseResult* FindLongOptParseResult(const TString& name, bool includeDefault = false) const;

        /**
         * Search for TOptParseResult that corresponds to given short name
         *
         * @param c                  short name of required object
         * @param includeDefault     search in results obtained from default values
         *
         * @return                   ptr on result
         */
        const TOptParseResult* FindCharOptParseResult(char c, bool includeDefault = false) const;

        /**
         * @return argv[0]
         */
        TString GetProgramName() const;

        /**
         * Print usage string.
         */
        void PrintUsage(IOutputStream& os = Cout) const;

        /**
         * @return position in [premuted argv] of the first free argument
         */
        size_t GetFreeArgsPos() const;

        /**
         * @return number of fetched free arguments
         */
        size_t GetFreeArgCount() const;

        /**
         * @return all fetched free arguments
         */
        TVector<TString> GetFreeArgs() const;

        /**
         * @return true if given option exist in results of parsing
         *
         * @param opt                ptr on required object
         * @param includeDefault     search in results obtained from default values
         *
         */
        bool Has(const TOpt* opt, bool includeDefault = false) const;

        /**
         * @return nil terminated string on the last fetched argument of given option
         *
         * @param opt                ptr on required object
         * @param includeDefault     search in results obtained from default values
         */
        const char* Get(const TOpt* opt, bool includeDefault = true) const;

        /**
         * @return nil terminated string on the last fetched argument of given option
         *    if option haven't been fetched, given defaultValue will be returned
         *
         * @param opt                ptr on required object
         * @param defaultValue
         */
        const char* GetOrElse(const TOpt* opt, const char* defaultValue) const;

        /**
         * @return true if given option exist in results of parsing
         *
         * @param name               long name of required object
         * @param includeDefault     search in results obtained from default values
         *
         */
        bool Has(const TString& name, bool includeDefault = false) const;

        /**
         * @return nil terminated string on the last fetched argument of given option
         *
         * @param name               long name of required object
         * @param includeDefault     search in results obtained from default values
         */
        const char* Get(const TString& name, bool includeDefault = true) const;

        /**
         * @return nil terminated string on the last fetched argument of given option
         *    if option haven't been fetched, given defaultValue will be returned
         *
         * @param name               long name of required object
         * @param defaultValue
         */
        const char* GetOrElse(const TString& name, const char* defaultValue) const;

        /**
         * @return true if given option exist in results of parsing
         *
         * @param c                  short name of required object
         * @param includeDefault     search in results obtained from default values
         *
         */
        bool Has(char name, bool includeDefault = false) const;

        /**
         * @return nil terminated string on the last fetched argument of given option
         *
         * @param c                  short name of required object
         * @param includeDefault     search in results obtained from default values
         */
        const char* Get(char name, bool includeDefault = true) const;

        /**
         * @return nil terminated string on the last fetched argument of given option
         *    if option haven't been fetched, given defaultValue will be returned
         *
         * @param c                  short name of required object
         * @param defaultValue
         */
        const char* GetOrElse(char name, const char* defaultValue) const;

        /**
         * for given option return parsed value of the last fetched argument
         * if option haven't been fetched, HandleError action is called
         *
         * @param opt       required option (one of: ptr, short name, long name).
         *
         * @return       FromString<T>(last feteched argument)
         */
        template <typename T, typename TKey>
        T Get(const TKey opt) const {
            const char* value = Get(opt);
            try {
                return NPrivate::OptFromString<T>(value, opt);
            } catch (...) {
                HandleError();
                throw;
            }
        }

        /**
         * for given option return parsed value of the last fetched argument
         * if option haven't been fetched, given defaultValue will be returned
         *
         * @param opt       required option (one of: ptr, short name, long name).
         * @param defaultValue
         *
         * @return       FromString<T>(last feteched argument)
         */
        template <typename T, typename TKey>
        T GetOrElse(const TKey opt, const T& defaultValue) const {
            if (Has(opt))
                return Get<T>(opt);
            else
                return defaultValue;
        }

        /**
         * @return returns the argv with which the parser was started
         */
        const char** GetSourceArgv() {
            return Parser_ ? Parser_->Argv_ : nullptr;
        }

        /**
         * @returns the argc with which the parser was started
         */
        size_t GetSourceArgc() {
            return Parser_ ? Parser_->Argc_ : 0;
        }
    };

    /**
     * NLastGetopt::TOptsParseResultException contains result of parsing argc,argv be parser.
     *
     * Unlike TOptsParseResult, if error during parsing occures, an exception is thrown.
     *
     */
    class TOptsParseResultException: public TOptsParseResult {
    public:
        TOptsParseResultException(const TOpts* options, int argc, const char* argv[]) {
            Init(options, argc, argv);
        }
        TOptsParseResultException(const TOpts* options, int argc, char* argv[]) {
            Init(options, argc, const_cast<const char**>(argv));
        }
        virtual ~TOptsParseResultException() = default;
        void HandleError() const override;

    protected:
        TOptsParseResultException() = default;
    };

}
