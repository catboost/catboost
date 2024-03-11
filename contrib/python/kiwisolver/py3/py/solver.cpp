/*-----------------------------------------------------------------------------
| Copyright (c) 2013-2019, Nucleic Development Team.
|
| Distributed under the terms of the Modified BSD License.
|
| The full license is in the file LICENSE, distributed with this software.
|----------------------------------------------------------------------------*/
#include <cppy/cppy.h>
#include <kiwi/kiwi.h>
#include "types.h"
#include "util.h"


namespace kiwisolver
{

namespace
{

PyObject*
Solver_new( PyTypeObject* type, PyObject* args, PyObject* kwargs )
{
	if( PyTuple_GET_SIZE( args ) != 0 || ( kwargs && PyDict_Size( kwargs ) != 0 ) )
		return cppy::type_error( "Solver.__new__ takes no arguments" );
	PyObject* pysolver = PyType_GenericNew( type, args, kwargs );
	if( !pysolver )
		return 0;
	Solver* self = reinterpret_cast<Solver*>( pysolver );
	new( &self->solver ) kiwi::Solver();
	return pysolver;
}


void
Solver_dealloc( Solver* self )
{
	self->solver.~Solver();
	Py_TYPE( self )->tp_free( pyobject_cast( self ) );
}


PyObject*
Solver_addConstraint( Solver* self, PyObject* other )
{
	if( !Constraint::TypeCheck( other ) )
		return cppy::type_error( other, "Constraint" );
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


PyObject*
Solver_removeConstraint( Solver* self, PyObject* other )
{
	if( !Constraint::TypeCheck( other ) )
		return cppy::type_error( other, "Constraint" );
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


PyObject*
Solver_hasConstraint( Solver* self, PyObject* other )
{
	if( !Constraint::TypeCheck( other ) )
		return cppy::type_error( other, "Constraint" );
	Constraint* cn = reinterpret_cast<Constraint*>( other );
	return cppy::incref( self->solver.hasConstraint( cn->constraint ) ? Py_True : Py_False );
}


PyObject*
Solver_addEditVariable( Solver* self, PyObject* args )
{
	PyObject* pyvar;
	PyObject* pystrength;
	if( !PyArg_ParseTuple( args, "OO", &pyvar, &pystrength ) )
		return 0;
	if( !Variable::TypeCheck( pyvar ) )
		return cppy::type_error( pyvar, "Variable" );
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


PyObject*
Solver_removeEditVariable( Solver* self, PyObject* other )
{
	if( !Variable::TypeCheck( other ) )
		return cppy::type_error( other, "Variable" );
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


PyObject*
Solver_hasEditVariable( Solver* self, PyObject* other )
{
	if( !Variable::TypeCheck( other ) )
		return cppy::type_error( other, "Variable" );
	Variable* var = reinterpret_cast<Variable*>( other );
	return cppy::incref( self->solver.hasEditVariable( var->variable ) ? Py_True : Py_False );
}


PyObject*
Solver_suggestValue( Solver* self, PyObject* args )
{
	PyObject* pyvar;
	PyObject* pyvalue;
	if( !PyArg_ParseTuple( args, "OO", &pyvar, &pyvalue ) )
		return 0;
	if( !Variable::TypeCheck( pyvar ) )
		return cppy::type_error( pyvar, "Variable" );
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


PyObject*
Solver_updateVariables( Solver* self )
{
	self->solver.updateVariables();
	Py_RETURN_NONE;
}


PyObject*
Solver_reset( Solver* self )
{
	self->solver.reset();
	Py_RETURN_NONE;
}


PyObject*
Solver_dump( Solver* self )
{
	cppy::ptr dump_str( PyUnicode_FromString( self->solver.dumps().c_str() ) );
	PyObject_Print( dump_str.get(), stdout, 0 );
	Py_RETURN_NONE;
}

PyObject*
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


static PyType_Slot Solver_Type_slots[] = {
    { Py_tp_dealloc, void_cast( Solver_dealloc ) },      /* tp_dealloc */
    { Py_tp_methods, void_cast( Solver_methods ) },      /* tp_methods */
    { Py_tp_new, void_cast( Solver_new ) },              /* tp_new */
    { Py_tp_alloc, void_cast( PyType_GenericAlloc ) },   /* tp_alloc */
    { Py_tp_free, void_cast( PyObject_Del ) },           /* tp_free */
    { 0, 0 },
};


} // namespace


// Initialize static variables (otherwise the compiler eliminates them)
PyTypeObject* Solver::TypeObject = NULL;


PyType_Spec Solver::TypeObject_Spec = {
	"kiwisolver.Solver",             /* tp_name */
	sizeof( Solver ),                /* tp_basicsize */
	0,                                   /* tp_itemsize */
	Py_TPFLAGS_DEFAULT|
    Py_TPFLAGS_BASETYPE,                 /* tp_flags */
    Solver_Type_slots                /* slots */
};


bool Solver::Ready()
{
    // The reference will be handled by the module to which we will add the type
	TypeObject = pytype_cast( PyType_FromSpec( &TypeObject_Spec ) );
    if( !TypeObject )
    {
        return false;
    }
    return true;
}


PyObject* DuplicateConstraint;

PyObject* UnsatisfiableConstraint;

PyObject* UnknownConstraint;

PyObject* DuplicateEditVariable;

PyObject* UnknownEditVariable;

PyObject* BadRequiredStrength;


bool init_exceptions()
{
 	DuplicateConstraint = PyErr_NewException(
 		const_cast<char*>( "kiwisolver.DuplicateConstraint" ), 0, 0 );
 	if( !DuplicateConstraint )
    {
        return false;
    }

  	UnsatisfiableConstraint = PyErr_NewException(
  		const_cast<char*>( "kiwisolver.UnsatisfiableConstraint" ), 0, 0 );
 	if( !UnsatisfiableConstraint )
 	{
        return false;
    }

  	UnknownConstraint = PyErr_NewException(
  		const_cast<char*>( "kiwisolver.UnknownConstraint" ), 0, 0 );
 	if( !UnknownConstraint )
 	{
        return false;
    }

  	DuplicateEditVariable = PyErr_NewException(
  		const_cast<char*>( "kiwisolver.DuplicateEditVariable" ), 0, 0 );
 	if( !DuplicateEditVariable )
 	{
        return false;
    }

  	UnknownEditVariable = PyErr_NewException(
  		const_cast<char*>( "kiwisolver.UnknownEditVariable" ), 0, 0 );
 	if( !UnknownEditVariable )
 	{
        return false;
    }

  	BadRequiredStrength = PyErr_NewException(
  		const_cast<char*>( "kiwisolver.BadRequiredStrength" ), 0, 0 );
 	if( !BadRequiredStrength )
 	{
        return false;
    }

	return true;
}

}  // namespace
