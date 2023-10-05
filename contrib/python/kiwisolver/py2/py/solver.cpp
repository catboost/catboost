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
#include "types.h"
#include "util.h"


using namespace PythonHelpers;


static PyObject*
Solver_new( PyTypeObject* type, PyObject* args, PyObject* kwargs )
{
	if( PyTuple_GET_SIZE( args ) != 0 || ( kwargs && PyDict_Size( kwargs ) != 0 ) )
		return py_type_fail( "Solver.__new__ takes no arguments" );
	PyObject* pysolver = PyType_GenericNew( type, args, kwargs );
	if( !pysolver )
		return 0;
	Solver* self = reinterpret_cast<Solver*>( pysolver );
	new( &self->solver ) kiwi::Solver();
	return pysolver;
}


static void
Solver_dealloc( Solver* self )
{
	self->solver.~Solver();
	Py_TYPE( self )->tp_free( pyobject_cast( self ) );
}


static PyObject*
Solver_addConstraint( Solver* self, PyObject* other )
{
	if( !Constraint::TypeCheck( other ) )
		return py_expected_type_fail( other, "Constraint" );
	Constraint* cn = reinterpret_cast<Constraint*>( other );
	try
	{
		self->solver.addConstraint( cn->constraint );
	}
	catch( const kiwi::DuplicateConstraint& )
	{
		PyErr_SetObject( DuplicateConstraint, other );
		return 0;
	}
	catch( const kiwi::UnsatisfiableConstraint& )
	{
		PyErr_SetObject( UnsatisfiableConstraint, other );
		return 0;
	}
	Py_RETURN_NONE;
}


static PyObject*
Solver_removeConstraint( Solver* self, PyObject* other )
{
	if( !Constraint::TypeCheck( other ) )
		return py_expected_type_fail( other, "Constraint" );
	Constraint* cn = reinterpret_cast<Constraint*>( other );
	try
	{
		self->solver.removeConstraint( cn->constraint );
	}
	catch( const kiwi::UnknownConstraint& )
	{
		PyErr_SetObject( UnknownConstraint, other );
		return 0;
	}
	Py_RETURN_NONE;
}


static PyObject*
Solver_hasConstraint( Solver* self, PyObject* other )
{
	if( !Constraint::TypeCheck( other ) )
		return py_expected_type_fail( other, "Constraint" );
	Constraint* cn = reinterpret_cast<Constraint*>( other );
	return newref( self->solver.hasConstraint( cn->constraint ) ? Py_True : Py_False );
}


static PyObject*
Solver_addEditVariable( Solver* self, PyObject* args )
{
	PyObject* pyvar;
	PyObject* pystrength;
	if( !PyArg_ParseTuple( args, "OO", &pyvar, &pystrength ) )
		return 0;
	if( !Variable::TypeCheck( pyvar ) )
		return py_expected_type_fail( pyvar, "Variable" );
	double strength;
	if( !convert_to_strength( pystrength, strength ) )
		return 0;
	Variable* var = reinterpret_cast<Variable*>( pyvar );
	try
	{
		self->solver.addEditVariable( var->variable, strength );
	}
	catch( const kiwi::DuplicateEditVariable& )
	{
		PyErr_SetObject( DuplicateEditVariable, pyvar );
		return 0;
	}
	catch( const kiwi::BadRequiredStrength& e )
	{
		PyErr_SetString( BadRequiredStrength, e.what() );
		return 0;
	}
	Py_RETURN_NONE;
}


static PyObject*
Solver_removeEditVariable( Solver* self, PyObject* other )
{
	if( !Variable::TypeCheck( other ) )
		return py_expected_type_fail( other, "Variable" );
	Variable* var = reinterpret_cast<Variable*>( other );
	try
	{
		self->solver.removeEditVariable( var->variable );
	}
	catch( const kiwi::UnknownEditVariable& )
	{
		PyErr_SetObject( UnknownEditVariable, other );
		return 0;
	}
	Py_RETURN_NONE;
}


static PyObject*
Solver_hasEditVariable( Solver* self, PyObject* other )
{
	if( !Variable::TypeCheck( other ) )
		return py_expected_type_fail( other, "Variable" );
	Variable* var = reinterpret_cast<Variable*>( other );
	return newref( self->solver.hasEditVariable( var->variable ) ? Py_True : Py_False );
}


static PyObject*
Solver_suggestValue( Solver* self, PyObject* args )
{
	PyObject* pyvar;
	PyObject* pyvalue;
	if( !PyArg_ParseTuple( args, "OO", &pyvar, &pyvalue ) )
		return 0;
	if( !Variable::TypeCheck( pyvar ) )
		return py_expected_type_fail( pyvar, "Variable" );
	double value;
	if( !convert_to_double( pyvalue, value ) )
		return 0;
	Variable* var = reinterpret_cast<Variable*>( pyvar );
	try
	{
		self->solver.suggestValue( var->variable, value );
	}
	catch( const kiwi::UnknownEditVariable& )
	{
		PyErr_SetObject( UnknownEditVariable, pyvar );
		return 0;
	}
	Py_RETURN_NONE;
}


static PyObject*
Solver_updateVariables( Solver* self )
{
	self->solver.updateVariables();
	Py_RETURN_NONE;
}


static PyObject*
Solver_reset( Solver* self )
{
	self->solver.reset();
	Py_RETURN_NONE;
}


static PyObject*
Solver_dump( Solver* self )
{
	PyObjectPtr dump_str( PyUnicode_FromString( self->solver.dumps().c_str() ) );
	PyObject_Print( dump_str.get(), stdout, 0 );
	Py_RETURN_NONE;
}

static PyObject*
Solver_dumps( Solver* self )
{
	return PyUnicode_FromString( self->solver.dumps().c_str() );
}

static PyMethodDef
Solver_methods[] = {
	{ "addConstraint", ( PyCFunction )Solver_addConstraint, METH_O,
	  "Add a constraint to the solver." },
	{ "removeConstraint", ( PyCFunction )Solver_removeConstraint, METH_O,
	  "Remove a constraint from the solver." },
	{ "hasConstraint", ( PyCFunction )Solver_hasConstraint, METH_O,
	  "Check whether the solver contains a constraint." },
	{ "addEditVariable", ( PyCFunction )Solver_addEditVariable, METH_VARARGS,
	  "Add an edit variable to the solver." },
	{ "removeEditVariable", ( PyCFunction )Solver_removeEditVariable, METH_O,
	  "Remove an edit variable from the solver." },
	{ "hasEditVariable", ( PyCFunction )Solver_hasEditVariable, METH_O,
	  "Check whether the solver contains an edit variable." },
	{ "suggestValue", ( PyCFunction )Solver_suggestValue, METH_VARARGS,
	  "Suggest a desired value for an edit variable." },
	{ "updateVariables", ( PyCFunction )Solver_updateVariables, METH_NOARGS,
	  "Update the values of the solver variables." },
	{ "reset", ( PyCFunction )Solver_reset, METH_NOARGS,
	  "Reset the solver to the initial empty starting condition." },
	{ "dump", ( PyCFunction )Solver_dump, METH_NOARGS,
	  "Dump a representation of the solver internals to stdout." },
	{ "dumps", ( PyCFunction )Solver_dumps, METH_NOARGS,
	  "Dump a representation of the solver internals to a string." },
	{ 0 } // sentinel
};


PyTypeObject Solver_Type = {
	PyVarObject_HEAD_INIT( &PyType_Type, 0 )
	"kiwisolver.Solver",                    /* tp_name */
	sizeof( Solver ),                       /* tp_basicsize */
	0,                                      /* tp_itemsize */
	(destructor)Solver_dealloc,             /* tp_dealloc */
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
	(reprfunc)0,                            /* tp_repr */
	(PyNumberMethods*)0,                    /* tp_as_number */
	(PySequenceMethods*)0,                  /* tp_as_sequence */
	(PyMappingMethods*)0,                   /* tp_as_mapping */
	(hashfunc)0,                            /* tp_hash */
	(ternaryfunc)0,                         /* tp_call */
	(reprfunc)0,                            /* tp_str */
	(getattrofunc)0,                        /* tp_getattro */
	(setattrofunc)0,                        /* tp_setattro */
	(PyBufferProcs*)0,                      /* tp_as_buffer */
	Py_TPFLAGS_DEFAULT|Py_TPFLAGS_BASETYPE, /* tp_flags */
	0,                                      /* Documentation string */
	(traverseproc)0,                        /* tp_traverse */
	(inquiry)0,                             /* tp_clear */
	(richcmpfunc)0,                         /* tp_richcompare */
	0,                                      /* tp_weaklistoffset */
	(getiterfunc)0,                         /* tp_iter */
	(iternextfunc)0,                        /* tp_iternext */
	(struct PyMethodDef*)Solver_methods,    /* tp_methods */
	(struct PyMemberDef*)0,                 /* tp_members */
	0,                                      /* tp_getset */
	0,                                      /* tp_base */
	0,                                      /* tp_dict */
	(descrgetfunc)0,                        /* tp_descr_get */
	(descrsetfunc)0,                        /* tp_descr_set */
	0,                                      /* tp_dictoffset */
	(initproc)0,                            /* tp_init */
	(allocfunc)PyType_GenericAlloc,         /* tp_alloc */
	(newfunc)Solver_new,                    /* tp_new */
	(freefunc)PyObject_Del,                 /* tp_free */
	(inquiry)0,                             /* tp_is_gc */
	0,                                      /* tp_bases */
	0,                                      /* tp_mro */
	0,                                      /* tp_cache */
	0,                                      /* tp_subclasses */
	0,                                      /* tp_weaklist */
	(destructor)0                           /* tp_del */
};


PyObject* DuplicateConstraint;

PyObject* UnsatisfiableConstraint;

PyObject* UnknownConstraint;

PyObject* DuplicateEditVariable;

PyObject* UnknownEditVariable;

PyObject* BadRequiredStrength;


int import_solver()
{
 	DuplicateConstraint = PyErr_NewException(
 		const_cast<char*>( "kiwisolver.DuplicateConstraint" ), 0, 0 );
 	if( !DuplicateConstraint )
 		return -1;
  	UnsatisfiableConstraint = PyErr_NewException(
  		const_cast<char*>( "kiwisolver.UnsatisfiableConstraint" ), 0, 0 );
 	if( !UnsatisfiableConstraint )
 		return -1;
  	UnknownConstraint = PyErr_NewException(
  		const_cast<char*>( "kiwisolver.UnknownConstraint" ), 0, 0 );
 	if( !UnknownConstraint )
 		return -1;
  	DuplicateEditVariable = PyErr_NewException(
  		const_cast<char*>( "kiwisolver.DuplicateEditVariable" ), 0, 0 );
 	if( !DuplicateEditVariable )
 		return -1;
  	UnknownEditVariable = PyErr_NewException(
  		const_cast<char*>( "kiwisolver.UnknownEditVariable" ), 0, 0 );
 	if( !UnknownEditVariable )
 		return -1;
  	BadRequiredStrength = PyErr_NewException(
  		const_cast<char*>( "kiwisolver.BadRequiredStrength" ), 0, 0 );
 	if( !BadRequiredStrength )
 		return -1;
	return PyType_Ready( &Solver_Type );
}
