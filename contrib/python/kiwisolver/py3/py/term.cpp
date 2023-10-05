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
Term_new( PyTypeObject* type, PyObject* args, PyObject* kwargs )
{
	static const char *kwlist[] = { "variable", "coefficient", 0 };
	PyObject* pyvar;
	PyObject* pycoeff = 0;
	if( !PyArg_ParseTupleAndKeywords(
		args, kwargs, "O|O:__new__", const_cast<char**>( kwlist ),
		&pyvar, &pycoeff ) )
		return 0;
	if( !Variable::TypeCheck( pyvar ) )
		return py_expected_type_fail( pyvar, "Variable" );
	double coefficient = 1.0;
	if( pycoeff && !convert_to_double( pycoeff, coefficient ) )
		return 0;
	PyObject* pyterm = PyType_GenericNew( type, args, kwargs );
	if( !pyterm )
		return 0;
	Term* self = reinterpret_cast<Term*>( pyterm );
	self->variable = newref( pyvar );
	self->coefficient = coefficient;
	return pyterm;
}


static void
Term_clear( Term* self )
{
	Py_CLEAR( self->variable );
}


static int
Term_traverse( Term* self, visitproc visit, void* arg )
{
	Py_VISIT( self->variable );
	return 0;
}


static void
Term_dealloc( Term* self )
{
	PyObject_GC_UnTrack( self );
	Term_clear( self );
	Py_TYPE( self )->tp_free( pyobject_cast( self ) );
}


static PyObject*
Term_repr( Term* self )
{
	std::stringstream stream;
	stream << self->coefficient << " * ";
	stream << reinterpret_cast<Variable*>( self->variable )->variable.name();
	return FROM_STRING( stream.str().c_str() );
}


static PyObject*
Term_variable( Term* self )
{
	return newref( self->variable );
}


static PyObject*
Term_coefficient( Term* self )
{
	return PyFloat_FromDouble( self->coefficient );
}


static PyObject*
Term_value( Term* self )
{
	Variable* pyvar = reinterpret_cast<Variable*>( self->variable );
	return PyFloat_FromDouble( self->coefficient * pyvar->variable.value() );
}


static PyObject*
Term_add( PyObject* first, PyObject* second )
{
	return BinaryInvoke<BinaryAdd, Term>()( first, second );
}


static PyObject*
Term_sub( PyObject* first, PyObject* second )
{
	return BinaryInvoke<BinarySub, Term>()( first, second );
}


static PyObject*
Term_mul( PyObject* first, PyObject* second )
{
	return BinaryInvoke<BinaryMul, Term>()( first, second );
}


static PyObject*
Term_div( PyObject* first, PyObject* second )
{
	return BinaryInvoke<BinaryDiv, Term>()( first, second );
}


static PyObject*
Term_neg( PyObject* value )
{
	return UnaryInvoke<UnaryNeg, Term>()( value );
}


static PyObject*
Term_richcmp( PyObject* first, PyObject* second, int op )
{
	switch( op )
	{
		case Py_EQ:
			return BinaryInvoke<CmpEQ, Term>()( first, second );
		case Py_LE:
			return BinaryInvoke<CmpLE, Term>()( first, second );
		case Py_GE:
			return BinaryInvoke<CmpGE, Term>()( first, second );
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
Term_methods[] = {
	{ "variable", ( PyCFunction )Term_variable, METH_NOARGS,
	  "Get the variable for the term." },
	{ "coefficient", ( PyCFunction )Term_coefficient, METH_NOARGS,
	  "Get the coefficient for the term." },
	{ "value", ( PyCFunction )Term_value, METH_NOARGS,
	  "Get the value for the term." },
	{ 0 } // sentinel
};


static PyNumberMethods
Term_as_number = {
	(binaryfunc)Term_add,       /* nb_add */
	(binaryfunc)Term_sub,       /* nb_subtract */
	(binaryfunc)Term_mul,       /* nb_multiply */
#if PY_MAJOR_VERSION < 3
	(binaryfunc)Term_div,       /* nb_divide */
#endif
	0,                          /* nb_remainder */
	0,                          /* nb_divmod */
	0,                          /* nb_power */
	(unaryfunc)Term_neg,        /* nb_negative */
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
	(binaryfunc)Term_div,       /* nb_true_divide */
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


PyTypeObject Term_Type = {
	PyVarObject_HEAD_INIT( &PyType_Type, 0 )
	"kiwisolver.Term",                      /* tp_name */
	sizeof( Term ),                         /* tp_basicsize */
	0,                                      /* tp_itemsize */
	(destructor)Term_dealloc,               /* tp_dealloc */
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
	(reprfunc)Term_repr,                    /* tp_repr */
	(PyNumberMethods*)&Term_as_number,      /* tp_as_number */
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
	(traverseproc)Term_traverse,            /* tp_traverse */
	(inquiry)Term_clear,                    /* tp_clear */
	(richcmpfunc)Term_richcmp,              /* tp_richcompare */
	0,                                      /* tp_weaklistoffset */
	(getiterfunc)0,                         /* tp_iter */
	(iternextfunc)0,                        /* tp_iternext */
	(struct PyMethodDef*)Term_methods,      /* tp_methods */
	(struct PyMemberDef*)0,                 /* tp_members */
	0,                                      /* tp_getset */
	0,                                      /* tp_base */
	0,                                      /* tp_dict */
	(descrgetfunc)0,                        /* tp_descr_get */
	(descrsetfunc)0,                        /* tp_descr_set */
	0,                                      /* tp_dictoffset */
	(initproc)0,                            /* tp_init */
	(allocfunc)PyType_GenericAlloc,         /* tp_alloc */
	(newfunc)Term_new,                      /* tp_new */
	(freefunc)PyObject_GC_Del,              /* tp_free */
	(inquiry)0,                             /* tp_is_gc */
	0,                                      /* tp_bases */
	0,                                      /* tp_mro */
	0,                                      /* tp_cache */
	0,                                      /* tp_subclasses */
	0,                                      /* tp_weaklist */
	(destructor)0                           /* tp_del */
};


int import_term()
{
	return PyType_Ready( &Term_Type );
}
