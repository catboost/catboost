/*-----------------------------------------------------------------------------
| Copyright (c) 2013-2019, Nucleic Development Team.
|
| Distributed under the terms of the Modified BSD License.
|
| The full license is in the file LICENSE, distributed with this software.
|----------------------------------------------------------------------------*/
#include <sstream>
#include <cppy/cppy.h>
#include "symbolics.h"
#include "types.h"
#include "util.h"


namespace kiwisolver
{


namespace
{


PyObject*
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
		return cppy::type_error( pyvar, "Variable" );
	double coefficient = 1.0;
	if( pycoeff && !convert_to_double( pycoeff, coefficient ) )
		return 0;
	PyObject* pyterm = PyType_GenericNew( type, args, kwargs );
	if( !pyterm )
		return 0;
	Term* self = reinterpret_cast<Term*>( pyterm );
	self->variable = cppy::incref( pyvar );
	self->coefficient = coefficient;
	return pyterm;
}


void
Term_clear( Term* self )
{
	Py_CLEAR( self->variable );
}


int
Term_traverse( Term* self, visitproc visit, void* arg )
{
	Py_VISIT( self->variable );
#if PY_VERSION_HEX >= 0x03090000
    // This was not needed before Python 3.9 (Python issue 35810 and 40217)
    Py_VISIT(Py_TYPE(self));
#endif
	return 0;
}


void
Term_dealloc( Term* self )
{
	PyObject_GC_UnTrack( self );
	Term_clear( self );
	Py_TYPE( self )->tp_free( pyobject_cast( self ) );
}


PyObject*
Term_repr( Term* self )
{
	std::stringstream stream;
	stream << self->coefficient << " * ";
	stream << reinterpret_cast<Variable*>( self->variable )->variable.name();
	return PyUnicode_FromString( stream.str().c_str() );
}


PyObject*
Term_variable( Term* self )
{
	return cppy::incref( self->variable );
}


PyObject*
Term_coefficient( Term* self )
{
	return PyFloat_FromDouble( self->coefficient );
}


PyObject*
Term_value( Term* self )
{
	Variable* pyvar = reinterpret_cast<Variable*>( self->variable );
	return PyFloat_FromDouble( self->coefficient * pyvar->variable.value() );
}


PyObject*
Term_add( PyObject* first, PyObject* second )
{
	return BinaryInvoke<BinaryAdd, Term>()( first, second );
}


PyObject*
Term_sub( PyObject* first, PyObject* second )
{
	return BinaryInvoke<BinarySub, Term>()( first, second );
}


PyObject*
Term_mul( PyObject* first, PyObject* second )
{
	return BinaryInvoke<BinaryMul, Term>()( first, second );
}


PyObject*
Term_div( PyObject* first, PyObject* second )
{
	return BinaryInvoke<BinaryDiv, Term>()( first, second );
}


PyObject*
Term_neg( PyObject* value )
{
	return UnaryInvoke<UnaryNeg, Term>()( value );
}


PyObject*
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
		Py_TYPE( first )->tp_name,
		Py_TYPE( second )->tp_name
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


static PyType_Slot Term_Type_slots[] = {
    { Py_tp_dealloc, void_cast( Term_dealloc ) },      /* tp_dealloc */
    { Py_tp_traverse, void_cast( Term_traverse ) },    /* tp_traverse */
    { Py_tp_clear, void_cast( Term_clear ) },          /* tp_clear */
    { Py_tp_repr, void_cast( Term_repr ) },            /* tp_repr */
    { Py_tp_richcompare, void_cast( Term_richcmp ) },  /* tp_richcompare */
    { Py_tp_methods, void_cast( Term_methods ) },      /* tp_methods */
    { Py_tp_new, void_cast( Term_new ) },              /* tp_new */
    { Py_tp_alloc, void_cast( PyType_GenericAlloc ) }, /* tp_alloc */
    { Py_tp_free, void_cast( PyObject_GC_Del ) },      /* tp_free */
    { Py_nb_add, void_cast( Term_add ) },              /* nb_add */
    { Py_nb_subtract, void_cast( Term_sub ) },         /* nb_subatract */
    { Py_nb_multiply, void_cast( Term_mul ) },         /* nb_multiply */
    { Py_nb_negative, void_cast( Term_neg ) },         /* nb_negative */
    { Py_nb_true_divide, void_cast( Term_div ) },      /* nb_true_divide */
    { 0, 0 },
};


} // namespace


// Initialize static variables (otherwise the compiler eliminates them)
PyTypeObject* Term::TypeObject = NULL;


PyType_Spec Term::TypeObject_Spec = {
	"kiwisolver.Term",             /* tp_name */
	sizeof( Term ),                /* tp_basicsize */
	0,                                   /* tp_itemsize */
	Py_TPFLAGS_DEFAULT|
    Py_TPFLAGS_HAVE_GC|
    Py_TPFLAGS_BASETYPE,                 /* tp_flags */
    Term_Type_slots                /* slots */
};


bool Term::Ready()
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
