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
Expression_new( PyTypeObject* type, PyObject* args, PyObject* kwargs )
{
    static const char *kwlist[] = { "terms", "constant", 0 };
    PyObject* pyterms;
    PyObject* pyconstant = 0;
    if( !PyArg_ParseTupleAndKeywords(
        args, kwargs, "O|O:__new__", const_cast<char**>( kwlist ),
        &pyterms, &pyconstant ) )
        return 0;
    cppy::ptr terms( PySequence_Tuple( pyterms ) );
    if( !terms )
        return 0;
    Py_ssize_t end = PyTuple_GET_SIZE( terms.get() );
    for( Py_ssize_t i = 0; i < end; ++i )
    {
        PyObject* item = PyTuple_GET_ITEM( terms.get(), i );
        if( !Term::TypeCheck( item ) )
            return cppy::type_error( item, "Term" );
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


void
Expression_clear( Expression* self )
{
    Py_CLEAR( self->terms );
}


int
Expression_traverse( Expression* self, visitproc visit, void* arg )
{
    Py_VISIT( self->terms );
#if PY_VERSION_HEX >= 0x03090000
    // This was not needed before Python 3.9 (Python issue 35810 and 40217)
    Py_VISIT(Py_TYPE(self));
#endif
    return 0;
}


void
Expression_dealloc( Expression* self )
{
    PyObject_GC_UnTrack( self );
    Expression_clear( self );
    Py_TYPE( self )->tp_free( pyobject_cast( self ) );
}


PyObject*
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
    return PyUnicode_FromString( stream.str().c_str() );
}


PyObject*
Expression_terms( Expression* self )
{
    return cppy::incref( self->terms );
}


PyObject*
Expression_constant( Expression* self )
{
    return PyFloat_FromDouble( self->constant );
}


PyObject*
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


PyObject*
Expression_add( PyObject* first, PyObject* second )
{
    return BinaryInvoke<BinaryAdd, Expression>()( first, second );
}


PyObject*
Expression_sub( PyObject* first, PyObject* second )
{
    return BinaryInvoke<BinarySub, Expression>()( first, second );
}


PyObject*
Expression_mul( PyObject* first, PyObject* second )
{
    return BinaryInvoke<BinaryMul, Expression>()( first, second );
}


PyObject*
Expression_div( PyObject* first, PyObject* second )
{
    return BinaryInvoke<BinaryDiv, Expression>()( first, second );
}


PyObject*
Expression_neg( PyObject* value )
{
    return UnaryInvoke<UnaryNeg, Expression>()( value );
}


PyObject*
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
        Py_TYPE( first )->tp_name,
        Py_TYPE( second )->tp_name
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


static PyType_Slot Expression_Type_slots[] = {
    { Py_tp_dealloc, void_cast( Expression_dealloc ) },      /* tp_dealloc */
    { Py_tp_traverse, void_cast( Expression_traverse ) },    /* tp_traverse */
    { Py_tp_clear, void_cast( Expression_clear ) },          /* tp_clear */
    { Py_tp_repr, void_cast( Expression_repr ) },            /* tp_repr */
    { Py_tp_richcompare, void_cast( Expression_richcmp ) },  /* tp_richcompare */
    { Py_tp_methods, void_cast( Expression_methods ) },      /* tp_methods */
    { Py_tp_new, void_cast( Expression_new ) },              /* tp_new */
    { Py_tp_alloc, void_cast( PyType_GenericAlloc ) },       /* tp_alloc */
    { Py_tp_free, void_cast( PyObject_GC_Del ) },            /* tp_free */
    { Py_nb_add, void_cast( Expression_add ) },              /* nb_add */
    { Py_nb_subtract, void_cast( Expression_sub ) },         /* nb_sub */
    { Py_nb_multiply, void_cast( Expression_mul ) },         /* nb_mul */
    { Py_nb_negative, void_cast( Expression_neg ) },         /* nb_neg */
    { Py_nb_true_divide, void_cast( Expression_div ) },      /* nb_div */
    { 0, 0 },
};


} // namespace


// Initialize static variables (otherwise the compiler eliminates them)
PyTypeObject* Expression::TypeObject = NULL;


PyType_Spec Expression::TypeObject_Spec = {
	"kiwisolver.Expression",             /* tp_name */
	sizeof( Expression ),                /* tp_basicsize */
	0,                                   /* tp_itemsize */
	Py_TPFLAGS_DEFAULT|
    Py_TPFLAGS_HAVE_GC|
    Py_TPFLAGS_BASETYPE,                 /* tp_flags */
    Expression_Type_slots                /* slots */
};


bool Expression::Ready()
{
    // The reference will be handled by the module to which we will add the type
	TypeObject = pytype_cast( PyType_FromSpec( &TypeObject_Spec ) );
    if( !TypeObject )
    {
        return false;
    }
    return true;
}

}  // namesapce kiwisolver
