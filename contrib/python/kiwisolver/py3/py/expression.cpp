/*-----------------------------------------------------------------------------
| Copyright (c) 2013-2017, Nucleic Development Team.
|
| Distributed under the terms of the Modified BSD License.
|
| The full license is in the file COPYING.txt, distributed with this software.
|----------------------------------------------------------------------------*/
#include <sstream>
#include <Python.h>
#include "pythonhelpers.h"
#include "symbolics.h"
#include "types.h"
#include "util.h"


using namespace PythonHelpers;


static PyObject*
Expression_new( PyTypeObject* type, PyObject* args, PyObject* kwargs )
{
    static const char *kwlist[] = { "terms", "constant", 0 };
    PyObject* pyterms;
    PyObject* pyconstant = 0;
    if( !PyArg_ParseTupleAndKeywords(
        args, kwargs, "O|O:__new__", const_cast<char**>( kwlist ),
        &pyterms, &pyconstant ) )
        return 0;
    PyObjectPtr terms( PySequence_Tuple( pyterms ) );
    if( !terms )
        return 0;
    Py_ssize_t end = PyTuple_GET_SIZE( terms.get() );
    for( Py_ssize_t i = 0; i < end; ++i )
    {
        PyObject* item = PyTuple_GET_ITEM( terms.get(), i );
        if( !Term::TypeCheck( item ) )
            return py_expected_type_fail( item, "Term" );
    }
    double constant = 0.0;
    if( pyconstant && !convert_to_double( pyconstant, constant ) )
        return 0;
    PyObject* pyexpr = PyType_GenericNew( type, args, kwargs );
    if( !pyexpr )
        return 0;
    Expression* self = reinterpret_cast<Expression*>( pyexpr );
    self->terms = terms.release();
    self->constant = constant;
    return pyexpr;
}


static void
Expression_clear( Expression* self )
{
    Py_CLEAR( self->terms );
}


static int
Expression_traverse( Expression* self, visitproc visit, void* arg )
{
    Py_VISIT( self->terms );
    return 0;
}


static void
Expression_dealloc( Expression* self )
{
    PyObject_GC_UnTrack( self );
    Expression_clear( self );
    Py_TYPE( self )->tp_free( pyobject_cast( self ) );
}


static PyObject*
Expression_repr( Expression* self )
{
    std::stringstream stream;
    Py_ssize_t end = PyTuple_GET_SIZE( self->terms );
    for( Py_ssize_t i = 0; i < end; ++i )
    {
        PyObject* item = PyTuple_GET_ITEM( self->terms, i );
        Term* term = reinterpret_cast<Term*>( item );
        stream << term->coefficient << " * ";
        stream << reinterpret_cast<Variable*>( term->variable )->variable.name();
        stream << " + ";
    }
    stream << self->constant;
    return FROM_STRING( stream.str().c_str() );
}


static PyObject*
Expression_terms( Expression* self )
{
    return newref( self->terms );
}


static PyObject*
Expression_constant( Expression* self )
{
    return PyFloat_FromDouble( self->constant );
}


static PyObject*
Expression_value( Expression* self )
{
    double result = self->constant;
    Py_ssize_t size = PyTuple_GET_SIZE( self->terms );
    for( Py_ssize_t i = 0; i < size; ++i )
    {
        PyObject* item = PyTuple_GET_ITEM( self->terms, i );
        Term* term = reinterpret_cast<Term*>( item );
        Variable* pyvar = reinterpret_cast<Variable*>( term->variable );
        result += term->coefficient * pyvar->variable.value();
    }
    return PyFloat_FromDouble( result );
}


static PyObject*
Expression_add( PyObject* first, PyObject* second )
{
    return BinaryInvoke<BinaryAdd, Expression>()( first, second );
}


static PyObject*
Expression_sub( PyObject* first, PyObject* second )
{
    return BinaryInvoke<BinarySub, Expression>()( first, second );
}


static PyObject*
Expression_mul( PyObject* first, PyObject* second )
{
    return BinaryInvoke<BinaryMul, Expression>()( first, second );
}


static PyObject*
Expression_div( PyObject* first, PyObject* second )
{
    return BinaryInvoke<BinaryDiv, Expression>()( first, second );
}


static PyObject*
Expression_neg( PyObject* value )
{
    return UnaryInvoke<UnaryNeg, Expression>()( value );
}


static PyObject*
Expression_richcmp( PyObject* first, PyObject* second, int op )
{
    switch( op )
    {
        case Py_EQ:
            return BinaryInvoke<CmpEQ, Expression>()( first, second );
        case Py_LE:
            return BinaryInvoke<CmpLE, Expression>()( first, second );
        case Py_GE:
            return BinaryInvoke<CmpGE, Expression>()( first, second );
        default:
            break;
    }
    PyErr_Format(
        PyExc_TypeError,
        "unsupported operand type(s) for %s: "
        "'%.100s' and '%.100s'",
        pyop_str( op ),
        first->ob_type->tp_name,
        second->ob_type->tp_name
    );
    return 0;
}


static PyMethodDef
Expression_methods[] = {
    { "terms", ( PyCFunction )Expression_terms, METH_NOARGS,
      "Get the tuple of terms for the expression." },
    { "constant", ( PyCFunction )Expression_constant, METH_NOARGS,
      "Get the constant for the expression." },
    { "value", ( PyCFunction )Expression_value, METH_NOARGS,
      "Get the value for the expression." },
    { 0 } // sentinel
};


static PyNumberMethods
Expression_as_number = {
    (binaryfunc)Expression_add, /* nb_add */
    (binaryfunc)Expression_sub, /* nb_subtract */
    (binaryfunc)Expression_mul, /* nb_multiply */
#if PY_MAJOR_VERSION < 3
    (binaryfunc)Expression_div, /* nb_divide */
#endif
    0,                          /* nb_remainder */
    0,                          /* nb_divmod */
    0,                          /* nb_power */
    (unaryfunc)Expression_neg,  /* nb_negative */
    0,                          /* nb_positive */
    0,                          /* nb_absolute */
#if PY_MAJOR_VERSION >= 3
    0,                          /* nb_bool */
#else
    0,                          /* nb_nonzero */
#endif
    0,                          /* nb_invert */
    0,                          /* nb_lshift */
    0,                          /* nb_rshift */
    0,                          /* nb_and */
    0,                          /* nb_xor */
    (binaryfunc)0,              /* nb_or */
#if PY_MAJOR_VERSION < 3
    0,                          /* nb_coerce */
#endif
    0,                          /* nb_int */
#if PY_MAJOR_VERSION >= 3
    (void *)0,                  /* nb_reserved */
#else
    0,                          /* nb_long */
#endif
    0,                          /* nb_float */
#if PY_MAJOR_VERSION < 3
    0,                          /* nb_oct */
    0,                          /* nb_hex */
#endif
    0,                          /* nb_inplace_add */
    0,                          /* nb_inplace_subtract */
    0,                          /* nb_inplace_multiply */
#if PY_MAJOR_VERSION < 3
    0,                          /* nb_inplace_divide */
#endif
    0,                          /* nb_inplace_remainder */
    0,                          /* nb_inplace_power */
    0,                          /* nb_inplace_lshift */
    0,                          /* nb_inplace_rshift */
    0,                          /* nb_inplace_and */
    0,                          /* nb_inplace_xor */
    0,                          /* nb_inplace_or */
    (binaryfunc)0,              /* nb_floor_divide */
    (binaryfunc)Expression_div, /* nb_true_divide */
    0,                          /* nb_inplace_floor_divide */
    0,                          /* nb_inplace_true_divide */
#if PY_VERSION_HEX >= 0x02050000
    (unaryfunc)0,               /* nb_index */
#endif
#if PY_VERSION_HEX >= 0x03050000
    (binaryfunc)0,              /* nb_matrix_multiply */
    (binaryfunc)0,              /* nb_inplace_matrix_multiply */
#endif
};


PyTypeObject Expression_Type = {
    PyVarObject_HEAD_INIT( &PyType_Type, 0 )
    "kiwisolver.Expression",                /* tp_name */
    sizeof( Expression ),                   /* tp_basicsize */
    0,                                      /* tp_itemsize */
    (destructor)Expression_dealloc,         /* tp_dealloc */
    (printfunc)0,                           /* tp_print */
    (getattrfunc)0,                         /* tp_getattr */
    (setattrfunc)0,                         /* tp_setattr */
#if PY_VERSION_HEX >= 0x03050000
    ( PyAsyncMethods* )0,                   /* tp_as_async */
#elif PY_VERSION_HEX >= 0x03000000
    ( void* ) 0,                            /* tp_reserved */
#else
    ( cmpfunc )0,                           /* tp_compare */
#endif
    (reprfunc)Expression_repr,              /* tp_repr */
    (PyNumberMethods*)&Expression_as_number,/* tp_as_number */
    (PySequenceMethods*)0,                  /* tp_as_sequence */
    (PyMappingMethods*)0,                   /* tp_as_mapping */
    (hashfunc)0,                            /* tp_hash */
    (ternaryfunc)0,                         /* tp_call */
    (reprfunc)0,                            /* tp_str */
    (getattrofunc)0,                        /* tp_getattro */
    (setattrofunc)0,                        /* tp_setattro */
    (PyBufferProcs*)0,                      /* tp_as_buffer */
#if PY_MAJOR_VERSION >= 3
    Py_TPFLAGS_DEFAULT|Py_TPFLAGS_HAVE_GC|Py_TPFLAGS_BASETYPE,
#else
    Py_TPFLAGS_DEFAULT|Py_TPFLAGS_HAVE_GC|Py_TPFLAGS_BASETYPE|Py_TPFLAGS_CHECKTYPES, /* tp_flags */
#endif
    0,                                      /* Documentation string */
    (traverseproc)Expression_traverse,      /* tp_traverse */
    (inquiry)Expression_clear,              /* tp_clear */
    (richcmpfunc)Expression_richcmp,        /* tp_richcompare */
    0,                                      /* tp_weaklistoffset */
    (getiterfunc)0,                         /* tp_iter */
    (iternextfunc)0,                        /* tp_iternext */
    (struct PyMethodDef*)Expression_methods,/* tp_methods */
    (struct PyMemberDef*)0,                 /* tp_members */
    0,                                      /* tp_getset */
    0,                                      /* tp_base */
    0,                                      /* tp_dict */
    (descrgetfunc)0,                        /* tp_descr_get */
    (descrsetfunc)0,                        /* tp_descr_set */
    0,                                      /* tp_dictoffset */
    (initproc)0,                            /* tp_init */
    (allocfunc)PyType_GenericAlloc,         /* tp_alloc */
    (newfunc)Expression_new,                /* tp_new */
    (freefunc)PyObject_GC_Del,              /* tp_free */
    (inquiry)0,                             /* tp_is_gc */
    0,                                      /* tp_bases */
    0,                                      /* tp_mro */
    0,                                      /* tp_cache */
    0,                                      /* tp_subclasses */
    0,                                      /* tp_weaklist */
    (destructor)0                           /* tp_del */
};


int import_expression()
{
    return PyType_Ready( &Expression_Type );
}
