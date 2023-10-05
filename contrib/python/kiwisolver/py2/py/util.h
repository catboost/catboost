/*-----------------------------------------------------------------------------
| Copyright (c) 2013-2017, Nucleic Development Team.
|
| Distributed under the terms of the Modified BSD License.
|
| The full license is in the file COPYING.txt, distributed with this software.
|----------------------------------------------------------------------------*/
#pragma once
#include <map>
#include <string>
#include <Python.h>
#include <kiwi/kiwi.h>
#include "pythonhelpers.h"
#include "types.h"


inline bool
convert_to_double( PyObject* obj, double& out )
{
    if( PyFloat_Check( obj ) )
    {
        out = PyFloat_AS_DOUBLE( obj );
        return true;
    }
#if PY_MAJOR_VERSION < 3
    if( PyInt_Check( obj ) )
    {
        out = double( PyInt_AsLong( obj ) );
        return true;
    }
#endif
    if( PyLong_Check( obj ) )
    {
        out = PyLong_AsDouble( obj );
        if( out == -1.0 && PyErr_Occurred() )
            return false;
        return true;
    }
    PythonHelpers::py_expected_type_fail( obj, "float, int, or long" );
    return false;
}


inline bool
convert_pystr_to_str( PyObject* value, std::string& out )
{
#if PY_MAJOR_VERSION >= 3
    out = PyUnicode_AsUTF8( value );
#else
    if( PyUnicode_Check( value ) )
    {
        PythonHelpers::PyObjectPtr py_str( PyUnicode_AsUTF8String( value ) );
        if( !py_str )
             return false;  // LCOV_EXCL_LINE
        out = PyString_AS_STRING( py_str.get() );
    }
    else
        out = PyString_AS_STRING( value );
#endif
    return true;
}


inline bool
convert_to_strength( PyObject* value, double& out )
{
#if PY_MAJOR_VERSION >= 3
    if( PyUnicode_Check( value ) )
    {
#else
    if( PyString_Check( value ) | PyUnicode_Check( value ))
    {
#endif
        std::string str;
        if( !convert_pystr_to_str( value, str ) )
            return false;
        if( str == "required" )
            out = kiwi::strength::required;
        else if( str == "strong" )
            out = kiwi::strength::strong;
        else if( str == "medium" )
            out = kiwi::strength::medium;
        else if( str == "weak" )
            out = kiwi::strength::weak;
        else
        {
            PyErr_Format(
                PyExc_ValueError,
                "string strength must be 'required', 'strong', 'medium', "
                "or 'weak', not '%s'",
                str.c_str()
            );
            return false;
        }
        return true;
    }
    if( !convert_to_double( value, out ) )
        return false;
    return true;
}


inline bool
convert_to_relational_op( PyObject* value, kiwi::RelationalOperator& out )
{
#if PY_MAJOR_VERSION >= 3
    if( !PyUnicode_Check( value ) )
    {
        PythonHelpers::py_expected_type_fail( value, "unicode" );
        return false;
    }
#else
    if( !(PyString_Check( value ) | PyUnicode_Check( value ) ) )
    {
        PythonHelpers::py_expected_type_fail( value, "str or unicode" );
        return false;
    }
#endif
    std::string str;
    if( !convert_pystr_to_str( value, str ) )
        return false;
    if( str == "==" )
        out = kiwi::OP_EQ;
    else if( str == "<=" )
        out = kiwi::OP_LE;
    else if( str == ">=" )
        out = kiwi::OP_GE;
    else
    {
        PyErr_Format(
            PyExc_ValueError,
            "relational operator must be '==', '<=', or '>=', not '%s'",
            str.c_str()
        );
        return false;
    }
    return true;
}


inline PyObject*
make_terms( const std::map<PyObject*, double>& coeffs )
{
    typedef std::map<PyObject*, double>::const_iterator iter_t;
    PythonHelpers::PyObjectPtr terms( PyTuple_New( coeffs.size() ) );
    if( !terms )
        return 0;
    Py_ssize_t size = PyTuple_GET_SIZE( terms.get() );
    for( Py_ssize_t i = 0; i < size; ++i ) // zero tuple for safe early return
        PyTuple_SET_ITEM( terms.get(), i, 0 );
    Py_ssize_t i = 0;
    iter_t it = coeffs.begin();
    iter_t end = coeffs.end();
    for( ; it != end; ++it, ++i )
    {
        PyObject* pyterm = PyType_GenericNew( &Term_Type, 0, 0 );
        if( !pyterm )
            return 0;
        Term* term = reinterpret_cast<Term*>( pyterm );
        term->variable = PythonHelpers::newref( it->first );
        term->coefficient = it->second;
        PyTuple_SET_ITEM( terms.get(), i, pyterm );
    }
    return terms.release();
}


inline PyObject*
reduce_expression( PyObject* pyexpr )  // pyexpr must be an Expression
{
    Expression* expr = reinterpret_cast<Expression*>( pyexpr );
    std::map<PyObject*, double> coeffs;
    Py_ssize_t size = PyTuple_GET_SIZE( expr->terms );
    for( Py_ssize_t i = 0; i < size; ++i )
    {
        PyObject* item = PyTuple_GET_ITEM( expr->terms, i );
        Term* term = reinterpret_cast<Term*>( item );
        coeffs[ term->variable ] += term->coefficient;
    }
    PythonHelpers::PyObjectPtr terms( make_terms( coeffs ) );
    if( !terms )
        return 0;
    PyObject* pynewexpr = PyType_GenericNew( &Expression_Type, 0, 0 );
    if( !pynewexpr )
        return 0;
    Expression* newexpr = reinterpret_cast<Expression*>( pynewexpr );
    newexpr->terms = terms.release();
    newexpr->constant = expr->constant;
    return pynewexpr;
}


inline kiwi::Expression
convert_to_kiwi_expression( PyObject* pyexpr )  // pyexpr must be an Expression
{
    Expression* expr = reinterpret_cast<Expression*>( pyexpr );
    std::vector<kiwi::Term> kterms;
    Py_ssize_t size = PyTuple_GET_SIZE( expr->terms );
    for( Py_ssize_t i = 0; i < size; ++i )
    {
        PyObject* item = PyTuple_GET_ITEM( expr->terms, i );
        Term* term = reinterpret_cast<Term*>( item );
        Variable* var = reinterpret_cast<Variable*>( term->variable );
        kterms.push_back( kiwi::Term( var->variable, term->coefficient ) );
    }
    return kiwi::Expression( kterms, expr->constant );
}


inline const char*
pyop_str( int op )
{
    switch( op )
    {
        case Py_LT:
            return "<";
        case Py_LE:
            return "<=";
        case Py_EQ:
            return "==";
        case Py_NE:
            return "!=";
        case Py_GT:
            return ">";
        case Py_GE:
            return ">=";
        default:
            return "";
    }
}
