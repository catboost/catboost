/*-----------------------------------------------------------------------------
| Copyright (c) 2013-2019, Nucleic Development Team.
|
| Distributed under the terms of the Modified BSD License.
|
| The full license is in the file LICENSE, distributed with this software.
|----------------------------------------------------------------------------*/
#pragma once
#include <cppy/cppy.h>
#include "types.h"
#include "util.h"


namespace kiwisolver
{

template<typename Op, typename T>
struct UnaryInvoke
{
	PyObject* operator()( PyObject* value )
	{
		return Op()( reinterpret_cast<T*>( value ) );
	}
};


template<typename Op, typename T>
struct BinaryInvoke
{
	PyObject* operator()( PyObject* first, PyObject* second )
	{
		if( T::TypeCheck( first ) )
			return invoke<Normal>( reinterpret_cast<T*>( first ), second );
		return invoke<Reverse>( reinterpret_cast<T*>( second ), first );
	}

	struct Normal
	{
		template<typename U>
		PyObject* operator()( T* primary, U secondary )
		{
			return Op()( primary, secondary );
		}
	};

	struct Reverse
	{
		template<typename U>
		PyObject* operator()( T* primary, U secondary )
		{
			return Op()( secondary, primary );
		}
	};

	template<typename Invk>
	PyObject* invoke( T* primary, PyObject* secondary )
	{
		if( Expression::TypeCheck( secondary ) )
			return Invk()( primary, reinterpret_cast<Expression*>( secondary ) );
		if( Term::TypeCheck( secondary ) )
			return Invk()( primary, reinterpret_cast<Term*>( secondary ) );
		if( Variable::TypeCheck( secondary ) )
			return Invk()( primary, reinterpret_cast<Variable*>( secondary ) );
		if( PyFloat_Check( secondary ) )
			return Invk()( primary, PyFloat_AS_DOUBLE( secondary ) );
		if( PyLong_Check( secondary ) )
		{
			double v = PyLong_AsDouble( secondary );
			if( v == -1 && PyErr_Occurred() )
				return 0;
			return Invk()( primary, v );
		}
		Py_RETURN_NOTIMPLEMENTED;
	}
};


struct BinaryMul
{
	template<typename T, typename U>
	PyObject* operator()( T first, U second )
	{
		Py_RETURN_NOTIMPLEMENTED;
	}
};


template<> inline
PyObject* BinaryMul::operator()( Variable* first, double second )
{
	PyObject* pyterm = PyType_GenericNew( Term::TypeObject, 0, 0 );
	if( !pyterm )
		return 0;
	Term* term = reinterpret_cast<Term*>( pyterm );
	term->variable = cppy::incref( pyobject_cast( first ) );
	term->coefficient = second;
	return pyterm;
}


template<> inline
PyObject* BinaryMul::operator()( Term* first, double second )
{
	PyObject* pyterm = PyType_GenericNew( Term::TypeObject, 0, 0 );
	if( !pyterm )
		return 0;
	Term* term = reinterpret_cast<Term*>( pyterm );
	term->variable = cppy::incref( first->variable );
	term->coefficient = first->coefficient * second;
	return pyterm;
}


template<> inline
PyObject* BinaryMul::operator()( Expression* first, double second )
{
	cppy::ptr pyexpr( PyType_GenericNew( Expression::TypeObject, 0, 0 ) );
	if( !pyexpr )
		return 0;
	Expression* expr = reinterpret_cast<Expression*>( pyexpr.get() );
	cppy::ptr terms( PyTuple_New( PyTuple_GET_SIZE( first->terms ) ) );
	if( !terms )
		return 0;
	Py_ssize_t end = PyTuple_GET_SIZE( first->terms );
	for( Py_ssize_t i = 0; i < end; ++i )  // memset 0 for safe error return
		PyTuple_SET_ITEM( terms.get(), i, 0 );
	for( Py_ssize_t i = 0; i < end; ++i )
	{
		PyObject* item = PyTuple_GET_ITEM( first->terms, i );
		PyObject* term = BinaryMul()( reinterpret_cast<Term*>( item ), second );
		if( !term )
			return 0;
		PyTuple_SET_ITEM( terms.get(), i, term );
	}
	expr->terms = terms.release();
	expr->constant = first->constant * second;
	return pyexpr.release();
}


template<> inline
PyObject* BinaryMul::operator()( double first, Variable* second )
{
	return operator()( second, first );
}


template<> inline
PyObject* BinaryMul::operator()( double first, Term* second )
{
	return operator()( second, first );
}


template<> inline
PyObject* BinaryMul::operator()( double first, Expression* second )
{
	return operator()( second, first );
}


struct BinaryDiv
{
	template<typename T, typename U>
	PyObject* operator()( T first, U second )
	{
		Py_RETURN_NOTIMPLEMENTED;
	}
};


template<> inline
PyObject* BinaryDiv::operator()( Variable* first, double second )
{
	if( second == 0.0 )
	{
		PyErr_SetString( PyExc_ZeroDivisionError, "float division by zero" );
		return 0;
	}
	return BinaryMul()( first, 1.0 / second );
}


template<> inline
PyObject* BinaryDiv::operator()( Term* first, double second )
{
	if( second == 0.0 )
	{
		PyErr_SetString( PyExc_ZeroDivisionError, "float division by zero" );
		return 0;
	}
	return BinaryMul()( first, 1.0 / second );
}


template<> inline
PyObject* BinaryDiv::operator()( Expression* first, double second )
{
	if( second == 0.0 )
	{
		PyErr_SetString( PyExc_ZeroDivisionError, "float division by zero" );
		return 0;
	}
	return BinaryMul()( first, 1.0 / second );
}


struct UnaryNeg
{
	template<typename T>
	PyObject* operator()( T value )
	{
		Py_RETURN_NOTIMPLEMENTED;
	}
};


template<> inline
PyObject* UnaryNeg::operator()( Variable* value )
{
	return BinaryMul()( value, -1.0 );
}


template<> inline
PyObject* UnaryNeg::operator()( Term* value )
{
	return BinaryMul()( value, -1.0 );
}


template<> inline
PyObject* UnaryNeg::operator()( Expression* value )
{
	return BinaryMul()( value, -1.0 );
}


struct BinaryAdd
{
	template<typename T, typename U>
	PyObject* operator()( T first, U second )
	{
		Py_RETURN_NOTIMPLEMENTED;
	}
};


template<> inline
PyObject* BinaryAdd::operator()( Expression* first, Expression* second )
{
	cppy::ptr pyexpr( PyType_GenericNew( Expression::TypeObject, 0, 0 ) );
	if( !pyexpr )
		return 0;
	Expression* expr = reinterpret_cast<Expression*>( pyexpr.get() );
	expr->constant = first->constant + second->constant;
	expr->terms = PySequence_Concat( first->terms, second->terms );
	if( !expr->terms )
		return 0;
	return pyexpr.release();
}


template<> inline
PyObject* BinaryAdd::operator()( Expression* first, Term* second )
{
	cppy::ptr pyexpr( PyType_GenericNew( Expression::TypeObject, 0, 0 ) );
	if( !pyexpr )
		return 0;
	PyObject* terms = PyTuple_New( PyTuple_GET_SIZE( first->terms ) + 1 );
	if( !terms )
		return 0;
	Py_ssize_t end = PyTuple_GET_SIZE( first->terms );
	for( Py_ssize_t i = 0; i < end; ++i )
	{
		PyObject* item = PyTuple_GET_ITEM( first->terms, i );
		PyTuple_SET_ITEM( terms, i, cppy::incref( item ) );
	}
	PyTuple_SET_ITEM( terms, end, cppy::incref( pyobject_cast( second ) ) );
	Expression* expr = reinterpret_cast<Expression*>( pyexpr.get() );
	expr->terms = terms;
	expr->constant = first->constant;
	return pyexpr.release();
}


template<> inline
PyObject* BinaryAdd::operator()( Expression* first, Variable* second )
{
	cppy::ptr temp( BinaryMul()( second, 1.0 ) );
	if( !temp )
		return 0;
	return operator()( first, reinterpret_cast<Term*>( temp.get() ) );
}


template<> inline
PyObject* BinaryAdd::operator()( Expression* first, double second )
{
	cppy::ptr pyexpr( PyType_GenericNew( Expression::TypeObject, 0, 0 ) );
	if( !pyexpr )
		return 0;
	Expression* expr = reinterpret_cast<Expression*>( pyexpr.get() );
	expr->terms = cppy::incref( first->terms );
	expr->constant = first->constant + second;
	return pyexpr.release();
}


template<> inline
PyObject* BinaryAdd::operator()( Term* first, double second )
{
	cppy::ptr pyexpr( PyType_GenericNew( Expression::TypeObject, 0, 0 ) );
	if( !pyexpr )
		return 0;
	Expression* expr = reinterpret_cast<Expression*>( pyexpr.get() );
	expr->constant = second;
	expr->terms = PyTuple_Pack( 1, first );
	if( !expr->terms )
		return 0;
	return pyexpr.release();
}


template<> inline
PyObject* BinaryAdd::operator()( Term* first, Expression* second )
{
	return operator()( second, first );
}


template<> inline
PyObject* BinaryAdd::operator()( Term* first, Term* second )
{
	cppy::ptr pyexpr( PyType_GenericNew( Expression::TypeObject, 0, 0 ) );
	if( !pyexpr )
		return 0;
	Expression* expr = reinterpret_cast<Expression*>( pyexpr.get() );
	expr->constant = 0.0;
	expr->terms = PyTuple_Pack( 2, first, second );
	if( !expr->terms )
		return 0;
	return pyexpr.release();
}


template<> inline
PyObject* BinaryAdd::operator()( Term* first, Variable* second )
{
	cppy::ptr temp( BinaryMul()( second, 1.0 ) );
	if( !temp )
		return 0;
	return BinaryAdd()( first, reinterpret_cast<Term*>( temp.get() ) );
}


template<> inline
PyObject* BinaryAdd::operator()( Variable* first, double second )
{
	cppy::ptr temp( BinaryMul()( first, 1.0 ) );
	if( !temp )
		return 0;
	return operator()( reinterpret_cast<Term*>( temp.get() ), second );
}


template<> inline
PyObject* BinaryAdd::operator()( Variable* first, Variable* second )
{
	cppy::ptr temp( BinaryMul()( first, 1.0 ) );
	if( !temp )
		return 0;
	return operator()( reinterpret_cast<Term*>( temp.get() ), second );
}


template<> inline
PyObject* BinaryAdd::operator()( Variable* first, Term* second )
{
	cppy::ptr temp( BinaryMul()( first, 1.0 ) );
	if( !temp )
		return 0;
	return operator()( reinterpret_cast<Term*>( temp.get() ), second );
}


template<> inline
PyObject* BinaryAdd::operator()( Variable* first, Expression* second )
{
	cppy::ptr temp( BinaryMul()( first, 1.0 ) );
	if( !temp )
		return 0;
	return operator()( reinterpret_cast<Term*>( temp.get() ), second );
}


template<> inline
PyObject* BinaryAdd::operator()( double first, Variable* second )
{
	return operator()( second, first );
}


template<> inline
PyObject* BinaryAdd::operator()( double first, Term* second )
{
	return operator()( second, first );
}


template<> inline
PyObject* BinaryAdd::operator()( double first, Expression* second )
{
	return operator()( second, first );
}


struct BinarySub
{
	template<typename T, typename U>
	PyObject* operator()( T first, U second )
	{
		Py_RETURN_NOTIMPLEMENTED;
	}
};


template<> inline
PyObject* BinarySub::operator()( Variable* first, double second )
{
	return BinaryAdd()( first, -second );
}


template<> inline
PyObject* BinarySub::operator()( Variable* first, Variable* second )
{
	cppy::ptr temp( UnaryNeg()( second ) );
	if( !temp )
		return 0;
	return BinaryAdd()( first, reinterpret_cast<Term*>( temp.get() ) );
}


template<> inline
PyObject* BinarySub::operator()( Variable* first, Term* second )
{
	cppy::ptr temp( UnaryNeg()( second ) );
	if( !temp )
		return 0;
	return BinaryAdd()( first, reinterpret_cast<Term*>( temp.get() ) );
}


template<> inline
PyObject* BinarySub::operator()( Variable* first, Expression* second )
{
	cppy::ptr temp( UnaryNeg()( second ) );
	if( !temp )
		return 0;
	return BinaryAdd()( first, reinterpret_cast<Expression*>( temp.get() ) );
}


template<> inline
PyObject* BinarySub::operator()( Term* first, double second )
{
	return BinaryAdd()( first, -second );
}


template<> inline
PyObject* BinarySub::operator()( Term* first, Variable* second )
{
	cppy::ptr temp( UnaryNeg()( second ) );
	if( !temp )
		return 0;
	return BinaryAdd()( first, reinterpret_cast<Term*>( temp.get() ) );
}


template<> inline
PyObject* BinarySub::operator()( Term* first, Term* second )
{
	cppy::ptr temp( UnaryNeg()( second ) );
	if( !temp )
		return 0;
	return BinaryAdd()( first, reinterpret_cast<Term*>( temp.get() ) );
}


template<> inline
PyObject* BinarySub::operator()( Term* first, Expression* second )
{
	cppy::ptr temp( UnaryNeg()( second ) );
	if( !temp )
		return 0;
	return BinaryAdd()( first, reinterpret_cast<Expression*>( temp.get() ) );
}


template<> inline
PyObject* BinarySub::operator()( Expression* first, double second )
{
	return BinaryAdd()( first, -second );
}


template<> inline
PyObject* BinarySub::operator()( Expression* first, Variable* second )
{
	cppy::ptr temp( UnaryNeg()( second ) );
	if( !temp )
		return 0;
	return BinaryAdd()( first, reinterpret_cast<Term*>( temp.get() ) );
}


template<> inline
PyObject* BinarySub::operator()( Expression* first, Term* second )
{
	cppy::ptr temp( UnaryNeg()( second ) );
	if( !temp )
		return 0;
	return BinaryAdd()( first, reinterpret_cast<Term*>( temp.get() ) );
}


template<> inline
PyObject* BinarySub::operator()( Expression* first, Expression* second )
{
	cppy::ptr temp( UnaryNeg()( second ) );
	if( !temp )
		return 0;
	return BinaryAdd()( first, reinterpret_cast<Expression*>( temp.get() ) );
}


template<> inline
PyObject* BinarySub::operator()( double first, Variable* second )
{
	cppy::ptr temp( UnaryNeg()( second ) );
	if( !temp )
		return 0;
	return BinaryAdd()( first, reinterpret_cast<Term*>( temp.get() ) );
}


template<> inline
PyObject* BinarySub::operator()( double first, Term* second )
{
	cppy::ptr temp( UnaryNeg()( second ) );
	if( !temp )
		return 0;
	return BinaryAdd()( first, reinterpret_cast<Term*>( temp.get() ) );
}


template<> inline
PyObject* BinarySub::operator()( double first, Expression* second )
{
	cppy::ptr temp( UnaryNeg()( second ) );
	if( !temp )
		return 0;
	return BinaryAdd()( first, reinterpret_cast<Expression*>( temp.get() ) );
}


template<typename T, typename U>
PyObject* makecn( T first, U second, kiwi::RelationalOperator op )
{
	cppy::ptr pyexpr( BinarySub()( first, second ) );
	if( !pyexpr )
		return 0;
	cppy::ptr pycn( PyType_GenericNew( Constraint::TypeObject, 0, 0 ) );
	if( !pycn )
		return 0;
	Constraint* cn = reinterpret_cast<Constraint*>( pycn.get() );
	cn->expression = reduce_expression( pyexpr.get() );
	if( !cn->expression )
		return 0;
	kiwi::Expression expr( convert_to_kiwi_expression( cn->expression ) );
	new( &cn->constraint ) kiwi::Constraint( expr, op, kiwi::strength::required );
	return pycn.release();
}


struct CmpEQ
{
	template<typename T, typename U>
	PyObject* operator()( T first, U second )
	{
		return makecn( first, second, kiwi::OP_EQ );
	}
};


struct CmpLE
{
	template<typename T, typename U>
	PyObject* operator()( T first, U second )
	{
		return makecn( first, second, kiwi::OP_LE );
	}
};


struct CmpGE
{
	template<typename T, typename U>
	PyObject* operator()( T first, U second )
	{
		return makecn( first, second, kiwi::OP_GE );
	}
};


}  // namespace kiwisolver
