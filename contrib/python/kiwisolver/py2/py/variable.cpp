/*-----------------------------------------------------------------------------
| Copyright (c) 2013-2017, Nucleic Development Team.
|
| Distributed under the terms of the Modified BSD License.
|
| The full license is in the file COPYING.txt, distributed with this software.
|----------------------------------------------------------------------------*/
#include <Python.h>
#include <kiwi/kiwi.h>
#include "pythonhelpers.h"
#include "symbolics.h"
#include "types.h"
#include "util.h"


using namespace PythonHelpers;


static PyObject*
Variable_new( PyTypeObject* type, PyObject* args, PyObject* kwargs )
{
	static const char *kwlist[] = { "name", "context", 0 };
	PyObject* context = 0;
	PyObject* name = 0;

	if( !PyArg_ParseTupleAndKeywords(
		args, kwargs, "|OO:__new__", const_cast<char**>( kwlist ),
		&name, &context ) )
		return 0;

	PyObjectPtr pyvar( PyType_GenericNew( type, args, kwargs ) );
	if( !pyvar )
		return 0;

	Variable* self = reinterpret_cast<Variable*>( pyvar.get() );
	self->context = xnewref( context );

	if( name != 0 )
	{
#if PY_MAJOR_VERSION >= 3
		if( !PyUnicode_Check( name ) )
			return py_expected_type_fail( name, "unicode" );
#else
		if( !( PyString_Check( name ) | PyUnicode_Check( name ) ) )
		{
			return py_expected_type_fail( name, "str or unicode" );
		}
#endif
		std::string c_name;
		if( !convert_pystr_to_str(name, c_name) )
			return 0;  // LCOV_EXCL_LINE
		new( &self->variable ) kiwi::Variable( c_name );
	}
	else
	{
		new( &self->variable ) kiwi::Variable();
	}

	return pyvar.release();
}


static void
Variable_clear( Variable* self )
{
	Py_CLEAR( self->context );
}


static int
Variable_traverse( Variable* self, visitproc visit, void* arg )
{
	Py_VISIT( self->context );
	return 0;
}


static void
Variable_dealloc( Variable* self )
{
	PyObject_GC_UnTrack( self );
	Variable_clear( self );
	self->variable.~Variable();
	Py_TYPE( self )->tp_free( pyobject_cast( self ) );
}


static PyObject*
Variable_repr( Variable* self )
{
	return FROM_STRING( self->variable.name().c_str() );
}


static PyObject*
Variable_name( Variable* self )
{
	return FROM_STRING( self->variable.name().c_str() );
}


static PyObject*
Variable_setName( Variable* self, PyObject* pystr )
{
#if PY_MAJOR_VERSION >= 3
	if( !PyUnicode_Check( pystr ) )
		return py_expected_type_fail( pystr, "unicode" );
#else
   if( !(PyString_Check( pystr ) | PyUnicode_Check( pystr ) ) )
    {
        return py_expected_type_fail( pystr, "str or unicode" );
    }
#endif
   std::string str;
   if( !convert_pystr_to_str( pystr, str ) )
       return 0;
   self->variable.setName( str );
	Py_RETURN_NONE;
}


static PyObject*
Variable_context( Variable* self )
{
	if( self->context )
		return newref( self->context );
	Py_RETURN_NONE;
}


static PyObject*
Variable_setContext( Variable* self, PyObject* value )
{
	if( value != self->context )
	{
		PyObject* temp = self->context;
		self->context = newref( value );
		Py_XDECREF( temp );
	}
	Py_RETURN_NONE;
}


static PyObject*
Variable_value( Variable* self )
{
	return PyFloat_FromDouble( self->variable.value() );
}


static PyObject*
Variable_add( PyObject* first, PyObject* second )
{
	return BinaryInvoke<BinaryAdd, Variable>()( first, second );
}


static PyObject*
Variable_sub( PyObject* first, PyObject* second )
{
	return BinaryInvoke<BinarySub, Variable>()( first, second );
}


static PyObject*
Variable_mul( PyObject* first, PyObject* second )
{
	return BinaryInvoke<BinaryMul, Variable>()( first, second );
}


static PyObject*
Variable_div( PyObject* first, PyObject* second )
{
	return BinaryInvoke<BinaryDiv, Variable>()( first, second );
}


static PyObject*
Variable_neg( PyObject* value )
{
	return UnaryInvoke<UnaryNeg, Variable>()( value );
}


static PyObject*
Variable_richcmp( PyObject* first, PyObject* second, int op )
{
	switch( op )
	{
		case Py_EQ:
			return BinaryInvoke<CmpEQ, Variable>()( first, second );
		case Py_LE:
			return BinaryInvoke<CmpLE, Variable>()( first, second );
		case Py_GE:
			return BinaryInvoke<CmpGE, Variable>()( first, second );
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
Variable_methods[] = {
	{ "name", ( PyCFunction )Variable_name, METH_NOARGS,
	  "Get the name of the variable." },
	{ "setName", ( PyCFunction )Variable_setName, METH_O,
	  "Set the name of the variable." },
	{ "context", ( PyCFunction )Variable_context, METH_NOARGS,
	  "Get the context object associated with the variable." },
	{ "setContext", ( PyCFunction )Variable_setContext, METH_O,
	  "Set the context object associated with the variable." },
	{ "value", ( PyCFunction )Variable_value, METH_NOARGS,
	  "Get the current value of the variable." },
	{ 0 } // sentinel
};


static PyNumberMethods
Variable_as_number = {
	(binaryfunc)Variable_add,   /* nb_add */
	(binaryfunc)Variable_sub,   /* nb_subtract */
	(binaryfunc)Variable_mul,   /* nb_multiply */
#if PY_MAJOR_VERSION < 3
	(binaryfunc)Variable_div,   /* nb_divide */
#endif
	0,                          /* nb_remainder */
	0,                          /* nb_divmod */
	0,                          /* nb_power */
	(unaryfunc)Variable_neg,    /* nb_negative */
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
	(binaryfunc)Variable_div,   /* nb_true_divide */
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


PyTypeObject Variable_Type = {
	PyVarObject_HEAD_INIT( &PyType_Type, 0 )
	"kiwisolver.Variable",                  /* tp_name */
	sizeof( Variable ),                     /* tp_basicsize */
	0,                                      /* tp_itemsize */
	(destructor)Variable_dealloc,           /* tp_dealloc */
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
	(reprfunc)Variable_repr,                /* tp_repr */
	(PyNumberMethods*)&Variable_as_number,  /* tp_as_number */
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
	(traverseproc)Variable_traverse,        /* tp_traverse */
	(inquiry)Variable_clear,                /* tp_clear */
	(richcmpfunc)Variable_richcmp,          /* tp_richcompare */
	0,                                      /* tp_weaklistoffset */
	(getiterfunc)0,                         /* tp_iter */
	(iternextfunc)0,                        /* tp_iternext */
	(struct PyMethodDef*)Variable_methods,  /* tp_methods */
	(struct PyMemberDef*)0,                 /* tp_members */
	0,                                      /* tp_getset */
	0,                                      /* tp_base */
	0,                                      /* tp_dict */
	(descrgetfunc)0,                        /* tp_descr_get */
	(descrsetfunc)0,                        /* tp_descr_set */
	0,                                      /* tp_dictoffset */
	(initproc)0,                            /* tp_init */
	(allocfunc)PyType_GenericAlloc,         /* tp_alloc */
	(newfunc)Variable_new,                  /* tp_new */
	(freefunc)PyObject_GC_Del,              /* tp_free */
	(inquiry)0,                             /* tp_is_gc */
	0,                                      /* tp_bases */
	0,                                      /* tp_mro */
	0,                                      /* tp_cache */
	0,                                      /* tp_subclasses */
	0,                                      /* tp_weaklist */
	(destructor)0                           /* tp_del */
};


int import_variable()
{
	return PyType_Ready( &Variable_Type );
}
