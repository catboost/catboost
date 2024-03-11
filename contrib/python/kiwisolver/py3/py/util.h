/*-----------------------------------------------------------------------------
| Copyright (c) 2013-2019, Nucleic Development Team.
|
| Distributed under the terms of the Modified BSD License.
|
| The full license is in the file LICENSE, distributed with this software.
|----------------------------------------------------------------------------*/
#pragma once
#include <map>
#include <string>
#include <cppy/cppy.h>
#include <kiwi/kiwi.h>
#include "types.h"


namespace kiwisolver
{

inline bool
convert_to_double( PyObject* obj, double& out )
{
    if( PyFloat_Check( obj ) )
    {
        out = PyFloat_AS_DOUBLE( obj );
        return true;
    }
    if( PyLong_Check( obj ) )
    {
        out = PyLong_AsDouble( obj );
        if( out == -1.0 && PyErr_Occurred() )
            return false;
        return true;
    }
    cppy::type_error( obj, "float, int, or long" );
    return false;
}


inline bool
convert_pystr_to_str( PyObject* value, std::string& out )
{
    out = PyUnicode_AsUTF8( value );
    return true;
}


inline bool
convert_to_strength( PyObject* value, double& out )
{
    if( PyUnicode_Check( value ) )
    {
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
    if( !PyUnicode_Check( value ) )
    {
        cppy::type_error( value, "str" );
        return false;
    }
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
    cppy::ptr terms( PyTuple_New( coeffs.size() ) );
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
        PyObject* pyterm = PyType_GenericNew( Term::TypeObject, 0, 0 );
        if( !pyterm )
            return 0;
        Term* term = reinterpret_cast<Term*>( pyterm );
        term->variable = cppy::incref( it->first );
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
    cppy::ptr terms( make_terms( coeffs ) );
    if( !terms )
        return 0;
    PyObject* pynewexpr = PyType_GenericNew( Expression::TypeObject, 0, 0 );
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


}  // namespace kiwisolver
