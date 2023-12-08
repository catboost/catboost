/*-----------------------------------------------------------------------------
| Copyright (c) 2013-2019, Nucleic Development Team.
|
| Distributed under the terms of the Modified BSD License.
|
| The full license is in the file LICENSE, distributed with this software.
|----------------------------------------------------------------------------*/
#include <cppy/cppy.h>
#include <kiwi/kiwi.h>
#include "symbolics.h"
#include "types.h"
#include "util.h"


namespace kiwisolver
{


namespace
{


PyObject*
Variable_new( PyTypeObject* type, PyObject* args, PyObject* kwargs )
{
	static const char *kwlist[] = { "name", "context", 0 };
	PyObject* context = 0;
	PyObject* name = 0;

	if( !PyArg_ParseTupleAndKeywords(
		args, kwargs, "|OO:__new__", const_cast<char**>( kwlist ),
		&name, &context ) )
		return 0;

	cppy::ptr pyvar( PyType_GenericNew( type, args, kwargs ) );
	if( !pyvar )
		return 0;

	Variable* self = reinterpret_cast<Variable*>( pyvar.get() );
	self->context = cppy::xincref( context );

	if( name != 0 )
	{
		if( !PyUnicode_Check( name ) )
			return cppy::type_error( name, "str" );
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


void
Variable_clear( Variable* self )
{
	Py_CLEAR( self->context );
}


int
Variable_traverse( Variable* self, visitproc visit, void* arg )
{
	Py_VISIT( self->context );
#if PY_VERSION_HEX >= 0x03090000
    // This was not needed before Python 3.9 (Python issue 35810 and 40217)
    Py_VISIT(Py_TYPE(self));
#endif
	return 0;
}


void
Variable_dealloc( Variable* self )
{
	PyObject_GC_UnTrack( self );
	Variable_clear( self );
	self->variable.~Variable();
	Py_TYPE( self )->tp_free( pyobject_cast( self ) );
}


PyObject*
Variable_repr( Variable* self )
{
	return PyUnicode_FromString( self->variable.name().c_str() );
}


PyObject*
Variable_name( Variable* self )
{
	return PyUnicode_FromString( self->variable.name().c_str() );
}


PyObject*
Variable_setName( Variable* self, PyObject* pystr )
{
	if( !PyUnicode_Check( pystr ) )
		return cppy::type_error( pystr, "str" );
   std::string str;
   if( !convert_pystr_to_str( pystr, str ) )
       return 0;
   self->variable.setName( str );
	Py_RETURN_NONE;
}


PyObject*
Variable_context( Variable* self )
{
	if( self->context )
		return cppy::incref( self->context );
	Py_RETURN_NONE;
}


PyObject*
Variable_setContext( Variable* self, PyObject* value )
{
	if( value != self->context )
	{
		PyObject* temp = self->context;
		self->context = cppy::incref( value );
		Py_XDECREF( temp );
	}
	Py_RETURN_NONE;
}


PyObject*
Variable_value( Variable* self )
{
	return PyFloat_FromDouble( self->variable.value() );
}


PyObject*
Variable_add( PyObject* first, PyObject* second )
{
	return BinaryInvoke<BinaryAdd, Variable>()( first, second );
}


PyObject*
Variable_sub( PyObject* first, PyObject* second )
{
	return BinaryInvoke<BinarySub, Variable>()( first, second );
}


PyObject*
Variable_mul( PyObject* first, PyObject* second )
{
	return BinaryInvoke<BinaryMul, Variable>()( first, second );
}


PyObject*
Variable_div( PyObject* first, PyObject* second )
{
	return BinaryInvoke<BinaryDiv, Variable>()( first, second );
}


PyObject*
Variable_neg( PyObject* value )
{
	return UnaryInvoke<UnaryNeg, Variable>()( value );
}


PyObject*
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
		Py_TYPE( first )->tp_name,
		Py_TYPE( second )->tp_name
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


static PyType_Slot Variable_Type_slots[] = {
    { Py_tp_dealloc, void_cast( Variable_dealloc ) },      /* tp_dealloc */
    { Py_tp_traverse, void_cast( Variable_traverse ) },    /* tp_traverse */
    { Py_tp_clear, void_cast( Variable_clear ) },          /* tp_clear */
    { Py_tp_repr, void_cast( Variable_repr ) },            /* tp_repr */
    { Py_tp_richcompare, void_cast( Variable_richcmp ) },  /* tp_richcompare */
    { Py_tp_methods, void_cast( Variable_methods ) },      /* tp_methods */
    { Py_tp_new, void_cast( Variable_new ) },              /* tp_new */
    { Py_tp_alloc, void_cast( PyType_GenericAlloc ) },     /* tp_alloc */
    { Py_tp_free, void_cast( PyObject_GC_Del ) },          /* tp_free */
    { Py_nb_add, void_cast( Variable_add ) },              /* nb_add */
    { Py_nb_subtract, void_cast( Variable_sub ) },         /* nb_subtract */
    { Py_nb_multiply, void_cast( Variable_mul ) },         /* nb_multiply */
    { Py_nb_negative, void_cast( Variable_neg ) },         /* nb_negative */
    { Py_nb_true_divide, void_cast( Variable_div ) },      /* nb_true_divide */
    { 0, 0 },
};


} // namespace


// Initialize static variables (otherwise the compiler eliminates them)
PyTypeObject* Variable::TypeObject = NULL;


PyType_Spec Variable::TypeObject_Spec = {
	"kiwisolver.Variable",             /* tp_name */
	sizeof( Variable ),                /* tp_basicsize */
	0,                                 /* tp_itemsize */
	Py_TPFLAGS_DEFAULT|
    Py_TPFLAGS_HAVE_GC|
    Py_TPFLAGS_BASETYPE,               /* tp_flags */
    Variable_Type_slots                /* slots */
};


bool Variable::Ready()
{
    // The reference will be handled by the module to which we will add the type
	TypeObject = pytype_cast( PyType_FromSpec( &TypeObject_Spec ) );
    if( !TypeObject )
    {
        return false;
    }
    return true;
}

}  // namespace kiwisolver
