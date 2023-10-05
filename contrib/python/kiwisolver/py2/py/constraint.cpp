/*-----------------------------------------------------------------------------
| Copyright (c) 2013-2017, Nucleic Development Team.
|
| Distributed under the terms of the Modified BSD License.
|
| The full license is in the file COPYING.txt, distributed with this software.
|----------------------------------------------------------------------------*/
#include <algorithm>
#include <sstream>
#include <Python.h>
#include <kiwi/kiwi.h>
#include "pythonhelpers.h"
#include "types.h"
#include "util.h"


using namespace PythonHelpers;


static PyObject*
Constraint_new( PyTypeObject* type, PyObject* args, PyObject* kwargs )
{
    static const char *kwlist[] = { "expression", "op", "strength", 0 };
    PyObject* pyexpr;
    PyObject* pyop;
    PyObject* pystrength = 0;
    if( !PyArg_ParseTupleAndKeywords(
        args, kwargs, "OO|O:__new__", const_cast<char**>( kwlist ),
        &pyexpr, &pyop, &pystrength ) )
        return 0;
    if( !Expression::TypeCheck( pyexpr ) )
        return py_expected_type_fail( pyexpr, "Expression" );
    kiwi::RelationalOperator op;
    if( !convert_to_relational_op( pyop, op ) )
        return 0;
    double strength = kiwi::strength::required;
    if( pystrength && !convert_to_strength( pystrength, strength ) )
        return 0;
    PyObjectPtr pycn( PyType_GenericNew( type, args, kwargs ) );
    if( !pycn )
        return 0;
    Constraint* cn = reinterpret_cast<Constraint*>( pycn.get() );
    cn->expression = reduce_expression( pyexpr );
    if( !cn->expression )
        return 0;
    kiwi::Expression expr( convert_to_kiwi_expression( cn->expression ) );
    new( &cn->constraint ) kiwi::Constraint( expr, op, strength );
    return pycn.release();
}


static void
Constraint_clear( Constraint* self )
{
    Py_CLEAR( self->expression );
}


static int
Constraint_traverse( Constraint* self, visitproc visit, void* arg )
{
    Py_VISIT( self->expression );
    return 0;
}


static void
Constraint_dealloc( Constraint* self )
{
    PyObject_GC_UnTrack( self );
    Constraint_clear( self );
    self->constraint.~Constraint();
    Py_TYPE( self )->tp_free( pyobject_cast( self ) );
}


static PyObject*
Constraint_repr( Constraint* self )
{
    std::stringstream stream;
    Expression* expr = reinterpret_cast<Expression*>( self->expression );
    Py_ssize_t size = PyTuple_GET_SIZE( expr->terms );
    for( Py_ssize_t i = 0; i < size; ++i )
    {
        PyObject* item = PyTuple_GET_ITEM( expr->terms, i );
        Term* term = reinterpret_cast<Term*>( item );
        stream << term->coefficient << " * ";
        stream << reinterpret_cast<Variable*>( term->variable )->variable.name();
        stream << " + ";
    }
    stream << expr->constant;
    switch( self->constraint.op() )
    {
        case kiwi::OP_EQ:
            stream << " == 0";
            break;
        case kiwi::OP_LE:
            stream << " <= 0";
            break;
        case kiwi::OP_GE:
            stream << " >= 0";
            break;
    }
    stream << " | strength = " << self->constraint.strength();
    return FROM_STRING( stream.str().c_str() );
}


static PyObject*
Constraint_expression( Constraint* self )
{
    return newref( self->expression );
}


static PyObject*
Constraint_op( Constraint* self )
{
    PyObject* res = 0;
    switch( self->constraint.op() )
    {
        case kiwi::OP_EQ:
            res = FROM_STRING( "==" );
            break;
        case kiwi::OP_LE:
            res = FROM_STRING( "<=" );
            break;
        case kiwi::OP_GE:
            res = FROM_STRING( ">=" );
            break;
    }
    return res;
}


static PyObject*
Constraint_strength( Constraint* self )
{
    return PyFloat_FromDouble( self->constraint.strength() );
}


static PyObject*
Constraint_or( PyObject* pyoldcn, PyObject* value )
{
    if( !Constraint::TypeCheck( pyoldcn ) )
        std::swap( pyoldcn, value );
    double strength;
    if( !convert_to_strength( value, strength ) )
        return 0;
    PyObject* pynewcn = PyType_GenericNew( &Constraint_Type, 0, 0 );
    if( !pynewcn )
        return 0;
    Constraint* oldcn = reinterpret_cast<Constraint*>( pyoldcn );
    Constraint* newcn = reinterpret_cast<Constraint*>( pynewcn );
    newcn->expression = newref( oldcn->expression );
    new( &newcn->constraint ) kiwi::Constraint( oldcn->constraint, strength );
    return pynewcn;
}


static PyMethodDef
Constraint_methods[] = {
    { "expression", ( PyCFunction )Constraint_expression, METH_NOARGS,
      "Get the expression object for the constraint." },
    { "op", ( PyCFunction )Constraint_op, METH_NOARGS,
      "Get the relational operator for the constraint." },
    { "strength", ( PyCFunction )Constraint_strength, METH_NOARGS,
      "Get the strength for the constraint." },
    { 0 } // sentinel
};


static PyNumberMethods
Constraint_as_number = {
    0,                          /* nb_add */
    0,                          /* nb_subtract */
    0,                          /* nb_multiply */
#if PY_MAJOR_VERSION < 3
    0,                          /* nb_divide */
#endif
    0,                          /* nb_remainder */
    0,                          /* nb_divmod */
    0,                          /* nb_power */
    0,                          /* nb_negative */
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
    (binaryfunc)Constraint_or,  /* nb_or */
#if PY_MAJOR_VERSION < 3
    0,                          /* nb_coerce */
#endif
    0,                          /* nb_int */
    0,                          /* nb_long */
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
    (binaryfunc)0,              /* nb_true_divide */
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


PyTypeObject Constraint_Type = {
    PyVarObject_HEAD_INIT( &PyType_Type, 0 )
    "kiwisolver.Constraint",                /* tp_name */
    sizeof( Constraint ),                   /* tp_basicsize */
    0,                                      /* tp_itemsize */
    (destructor)Constraint_dealloc,         /* tp_dealloc */
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
    (reprfunc)Constraint_repr,              /* tp_repr */
    (PyNumberMethods*)&Constraint_as_number,/* tp_as_number */
    (PySequenceMethods*)0,                  /* tp_as_sequence */
    (PyMappingMethods*)0,                   /* tp_as_mapping */
    (hashfunc)0,                            /* tp_hash */
    (ternaryfunc)0,                         /* tp_call */
    (reprfunc)0,                            /* tp_str */
    (getattrofunc)0,                        /* tp_getattro */
    (setattrofunc)0,                        /* tp_setattro */
    (PyBufferProcs*)0,                      /* tp_as_buffer */
#if PY_MAJOR_VERSION >= 3
    Py_TPFLAGS_DEFAULT|Py_TPFLAGS_HAVE_GC|Py_TPFLAGS_BASETYPE, /* tp_flags */
#else
    Py_TPFLAGS_DEFAULT|Py_TPFLAGS_HAVE_GC|Py_TPFLAGS_BASETYPE|Py_TPFLAGS_CHECKTYPES, /* tp_flags */
#endif
    0,                                      /* Documentation string */
    (traverseproc)Constraint_traverse,      /* tp_traverse */
    (inquiry)Constraint_clear,              /* tp_clear */
    (richcmpfunc)0,                         /* tp_richcompare */
    0,                                      /* tp_weaklistoffset */
    (getiterfunc)0,                         /* tp_iter */
    (iternextfunc)0,                        /* tp_iternext */
    (struct PyMethodDef*)Constraint_methods,/* tp_methods */
    (struct PyMemberDef*)0,                 /* tp_members */
    0,                                      /* tp_getset */
    0,                                      /* tp_base */
    0,                                      /* tp_dict */
    (descrgetfunc)0,                        /* tp_descr_get */
    (descrsetfunc)0,                        /* tp_descr_set */
    0,                                      /* tp_dictoffset */
    (initproc)0,                            /* tp_init */
    (allocfunc)PyType_GenericAlloc,         /* tp_alloc */
    (newfunc)Constraint_new,                /* tp_new */
    (freefunc)PyObject_GC_Del,              /* tp_free */
    (inquiry)0,                             /* tp_is_gc */
    0,                                      /* tp_bases */
    0,                                      /* tp_mro */
    0,                                      /* tp_cache */
    0,                                      /* tp_subclasses */
    0,                                      /* tp_weaklist */
    (destructor)0                           /* tp_del */
};


int import_constraint()
{
    return PyType_Ready( &Constraint_Type );
}
