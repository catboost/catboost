#pragma once

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <util/generic/yexception.h>
#include <util/generic/map.h>
#include <util/generic/vector.h>
#include "ptr.h"

namespace NPyBind {
    // Usage:
    //   ythrow TPyErr(PyExc_TypeError) << "some python type error somewhere in your C++ code";
    //
    class TPyErr: public virtual yexception {
    public:
        TPyErr(PyObject* theException = PyExc_RuntimeError)
            : Exception(theException)
        {
        }

        TPyObjectPtr GetException() const {
            return Exception;
        }

    private:
        NPyBind::TPyObjectPtr Exception;
    };

    // Throw it when you set py error but function return type is not PyObject*.
    // Look into examples for more details.
    struct TPyNativeErrorException final: public yexception {};

    //Private api for creating PyBind python module
    //Needs only for overriding pybind python module in library which imports other pybind library
    namespace NPrivate {
        TPyObjectPtr CreatePyBindModule();
    }//NPrivate
    class TExceptionsHolder {
        friend TPyObjectPtr NPrivate::CreatePyBindModule();
    private:
        TExceptionsHolder(const TExceptionsHolder&);
        TExceptionsHolder& operator=(const TExceptionsHolder&);
        TExceptionsHolder();

        void Clear();
        TPyObjectPtr GetException(const TString&);
        TPyObjectPtr GetExceptions(const TVector<TString>&);
    private:
        class TExceptionsChecker {
        public:
            virtual ~TExceptionsChecker() {
            }
            virtual bool Check(const std::exception& ex) const = 0;
            virtual TString GetName() const = 0;
            virtual TPyObjectPtr GetException() const = 0;
        };

        template <typename TExcType>
        class TConcreteExceptionsChecker: public TExceptionsChecker {
        private:
            TString Name;
            TPyObjectPtr Exception;

        public:
            TConcreteExceptionsChecker(const TString& name, TPyObjectPtr exception)
                : Name(name)
                , Exception(exception)
            {
            }

            bool Check(const std::exception& ex) const override {
                const std::exception* e = &ex;
                return dynamic_cast<const TExcType*>(e);
            }

            TString GetName() const override {
                return Name;
            }

            TPyObjectPtr GetException() const override {
                return Exception;
            }
        };

        class TPyErrExceptionsChecker: public TExceptionsChecker {
        private:
            mutable TPyObjectPtr Exception;

        public:
            TPyErrExceptionsChecker() {
            }

            bool Check(const std::exception& ex) const override {
                const TPyErr* err = dynamic_cast<const TPyErr*>(&ex);
                if (err) {
                    Exception = err->GetException();
                }
                return err != nullptr;
            }

            TString GetName() const override {
                return TString();
            }

            TPyObjectPtr GetException() const override {
                return Exception;
            }
        };

        typedef TSimpleSharedPtr<TExceptionsChecker> TCheckerPtr;
        typedef TVector<TCheckerPtr> TCheckersVector;
        typedef TMap<TString, TPyObjectPtr> TExceptionsMap;

        TPyObjectPtr Module;
        TCheckersVector Checkers;
        TExceptionsMap Exceptions;

        static PyObject* DoInitPyBindModule();
        static void DoInitPyBindModule2();

    public:
        static TExceptionsHolder& Instance() {
            static TExceptionsHolder Holder;
            return Holder;
        }

        template <typename TExcType>
        void AddException(const TString& name, const TString& base = "") {
            TPyObjectPtr baseException(GetException(base));
            TString fullName = TString("pybind.") + name;
            TPyObjectPtr exception(PyErr_NewException(const_cast<char*>(fullName.c_str()), baseException.Get(), nullptr), true);
            Checkers.push_back(new TConcreteExceptionsChecker<TExcType>(name, exception));
            Exceptions[name] = exception;
        }

        template <typename TExcType>
        void AddException(const TString& name, const TVector<TString>& bases) {
            TPyObjectPtr baseExceptions(GetExceptions(bases));
            TString fullName = TString("pybind.") + name;
            TPyObjectPtr exception(PyErr_NewException(const_cast<char*>(fullName.c_str()), baseExceptions.Get(), nullptr), true);
            Checkers.push_back(new TConcreteExceptionsChecker<TExcType>(name, exception));
            Exceptions[name] = exception;
        }

        NPyBind::TPyObjectPtr ToPyException(const std::exception&);
    };
}
