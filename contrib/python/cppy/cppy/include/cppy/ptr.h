/*-----------------------------------------------------------------------------
| Copyright (c) 2014-2020, Nucleic
|
| Distributed under the terms of the BSD 3-Clause License.
|
| The full license is in the file LICENSE, distributed with this software.
|----------------------------------------------------------------------------*/
#pragma once

#include <string>
#include <Python.h>
#include "defines.h"


namespace cppy
{

template <typename T>
inline T* incref( T* ob )
{
	Py_INCREF( ob );
	return ob;
}


template <typename T>
inline T* xincref( T* ob )
{
	Py_XINCREF( ob );
	return ob;
}


template <typename T>
inline T* decref( T* ob )
{
	Py_DECREF( ob );
	return ob;
}


template <typename T>
inline T* xdecref( T* ob )
{
	Py_XDECREF( ob );
	return ob;
}


template <typename T>
inline void clear( T** ob )
{
	T* temp = *ob;
	*ob = 0;
	Py_XDECREF( temp );
}


template <typename T>
inline void replace( T** src, T* ob )
{
	T* temp = *src;
	*src = ob;
	Py_XINCREF( ob );
	Py_XDECREF( temp );
}


class ptr
{

public:
	ptr() : m_ob( 0 )
	{
	}

	ptr( const ptr& other ) : m_ob( cppy::xincref( other.get() ) )
	{
	}

	ptr( PyObject* ob, bool incref = false )
		: m_ob( incref ? cppy::xincref( ob ) : ob )
	{
	}

	~ptr()
	{
		PyObject* temp = m_ob;
		m_ob = 0;
		Py_XDECREF( temp );
	}

	ptr& operator=( PyObject* other )
	{
		PyObject* temp = m_ob;
		m_ob = other;
		Py_XDECREF( temp );
		return *this;
	}

	ptr& operator=( const ptr& other )
	{
		PyObject* temp = m_ob;
		m_ob = other.get();
		Py_XINCREF( m_ob );
		Py_XDECREF( temp );
		return *this;
	}

	PyObject* get() const
	{
		return m_ob;
	}

	void set( PyObject* ob, bool incref = false )
	{
		PyObject* temp = m_ob;
		m_ob = incref ? cppy::xincref( ob ) : ob;
		Py_XDECREF( temp );
	}

	void set( const ptr& other )
	{
		PyObject* temp = m_ob;
		m_ob = other.get();
		Py_XINCREF( m_ob );
		Py_XDECREF( temp );
	}

	PyObject* release()
	{
		PyObject* temp = m_ob;
		m_ob = 0;
		return temp;
	}

	operator void*() const
	{
		return static_cast<void*>( m_ob );
	}

	bool is_none() const
	{
		return m_ob == Py_None;
	}

	bool is_true() const
	{
		return m_ob == Py_True;
	}

	bool is_false() const
	{
		return m_ob == Py_False;
	}

	bool is_bool() const
	{
		return is_true() || is_false();
	}

	bool is_int() const
	{
		return PyLong_Check( m_ob ) != 0;
	}

	bool is_float() const
	{
		return PyFloat_Check( m_ob ) != 0;
	}

	bool is_list() const
	{
		return PyList_Check( m_ob ) != 0;
	}

	bool is_dict() const
	{
		return PyDict_Check( m_ob ) != 0;
	}

	bool is_set() const
	{
		return PySet_Check( m_ob ) != 0;
	}

	bool is_bytes() const
	{
		return PyBytes_Check( m_ob ) != 0;
	}

	bool is_str() const
	{
		return PyUnicode_Check( m_ob ) != 0;
	}

	bool is_unicode() const
	{
		return PyUnicode_Check( m_ob ) != 0;
	}

	bool is_callable() const
	{
		return PyCallable_Check( m_ob ) != 0;
	}

	bool is_iter() const
	{
		return PyIter_Check( m_ob ) != 0;
	}

	bool is_type( PyTypeObject* cls ) const
	{
		return PyObject_TypeCheck( m_ob, cls ) != 0;
	}

	int is_truthy() const
	{
		return PyObject_IsTrue( m_ob );
	}

	int is_instance( PyObject* cls ) const
	{
		return PyObject_IsInstance( m_ob, cls );
	}

	int is_instance( const ptr& cls ) const
	{
		return is_instance( cls.get() );
	}

	int is_subclass( PyObject* cls ) const
	{
		return PyObject_IsSubclass( m_ob, cls );
	}

	int is_subclass( const ptr& cls ) const
	{
		return is_subclass( cls.get() );
	}

	PyObject* iter() const
	{
		return PyObject_GetIter( m_ob );
	}

	PyObject* next() const
	{
		return PyIter_Next( m_ob );
	}

	PyObject* repr() const
	{
		return PyObject_Repr( m_ob );
	}

	PyObject* str() const
	{
		return PyObject_Str( m_ob );
	}

	PyObject* bytes() const
	{
		return PyObject_Bytes( m_ob );
	}

	PyObject* unicode() const
	{
		return PyObject_Str( m_ob );
	}

	Py_ssize_t length() const
	{
		return PyObject_Length( m_ob );
	}

	PyTypeObject* type() const
	{
		return Py_TYPE( m_ob );
	}

	int richcmp( PyObject* other, int opid ) const
	{
		return PyObject_RichCompareBool( m_ob, other, opid );
	}

	int richcmp( const ptr& other, int opid ) const
	{
		return richcmp( other.get(), opid );
	}

	Py_hash_t hash() const
	{
		return PyObject_Hash( m_ob );
	}

	bool hasattr( PyObject* attr ) const
	{
		return PyObject_HasAttr( m_ob, attr ) == 1;
	}

	bool hasattr( const ptr& attr ) const
	{
		return hasattr( attr.get() );
	}

	bool hasattr( const char* attr ) const
	{
		return PyObject_HasAttrString( m_ob, attr ) == 1;
	}

	bool hasattr( const std::string& attr ) const
	{
		return hasattr( attr.c_str() );
	}

	PyObject* getattr( PyObject* attr ) const
	{
		return PyObject_GetAttr( m_ob, attr );
	}

	PyObject* getattr( const ptr& attr ) const
	{
		return getattr( attr.get() );
	}

	PyObject* getattr( const char* attr ) const
	{
		return PyObject_GetAttrString( m_ob, attr );
	}

	PyObject* getattr( const std::string& attr ) const
	{
		return getattr( attr.c_str() );
	}

	bool setattr( PyObject* attr, PyObject* value ) const
	{
		return PyObject_SetAttr( m_ob, attr, value ) == 0;
	}

	bool setattr( const ptr& attr, PyObject* value ) const
	{
		return setattr( attr.get(), value );
	}

	bool setattr( PyObject* attr, const ptr& value ) const
	{
		return setattr( attr, value.get() );
	}

	bool setattr( const ptr& attr, const ptr& value ) const
	{
		return setattr( attr.get(), value.get() );
	}

	bool setattr( const char* attr, PyObject* value ) const
	{
		return PyObject_SetAttrString( m_ob, attr, value ) == 0;
	}

	bool setattr( const char* attr, const ptr& value ) const
	{
		return setattr( attr, value.get() );
	}

	bool setattr( const std::string& attr, PyObject* value ) const
	{
		return setattr( attr.c_str(), value );
	}

	bool setattr( const std::string& attr, const ptr& value ) const
	{
		return setattr( attr.c_str(), value.get() );
	}

	bool delattr( PyObject* attr ) const
	{
		return PyObject_DelAttr( m_ob, attr ) == 0;
	}

	bool delattr( const ptr& attr ) const
	{
		return delattr( attr.get() );
	}

	bool delattr( const char* attr ) const
	{
		return PyObject_DelAttrString( m_ob, attr ) == 0;
	}

	bool delattr( const std::string& attr ) const
	{
		return delattr( attr.c_str() );
	}

	PyObject* getitem( PyObject* key ) const
	{
		return PyObject_GetItem( m_ob, key );
	}

	PyObject* getitem( const ptr& key ) const
	{
		return getitem( key.get() );
	}

	bool setitem( PyObject* key, PyObject* value ) const
	{
		return PyObject_SetItem( m_ob, key, value ) == 0;
	}

	bool setitem( const ptr& key, PyObject* value ) const
	{
		return setitem( key.get(), value );
	}

	bool setitem( PyObject* key, const ptr& value ) const
	{
		return setitem( key, value.get() );
	}

	bool setitem( const ptr& key, const ptr& value ) const
	{
		return setitem( key.get(), value.get() );
	}

	bool delitem( PyObject* key )
	{
		return PyObject_DelItem( m_ob, key ) == 0;
	}

	bool delitem( const ptr& key )
	{
		return delitem( key.get() );
	}

	PyObject* call( PyObject* args, PyObject* kwargs = 0 ) const
	{
		return PyObject_Call( m_ob, args, kwargs );
	}

	PyObject* call( const ptr& args ) const
	{
		return call( args.get() );
	}

	PyObject* call( const ptr& args, const ptr& kwargs ) const
	{
		return call( args.get(), kwargs.get() );
	}

	PyObject* call( const ptr& args, PyObject* kwargs ) const
	{
		return call( args.get(), kwargs );
	}

	PyObject* call( PyObject* args, const ptr& kwargs ) const
	{
		return call( args, kwargs.get() );
	}

protected:
	PyObject* m_ob;
};


inline bool operator!=( const ptr& lhs, const ptr& rhs )
{
	return lhs.get() != rhs.get();
}


inline bool operator!=( PyObject* lhs, const ptr& rhs )
{
	return lhs != rhs.get();
}


inline bool operator!=( const ptr& lhs, PyObject* rhs )
{
	return lhs.get() != rhs;
}


inline bool operator==( const ptr& lhs, const ptr& rhs )
{
	return lhs.get() == rhs.get();
}


inline bool operator==( PyObject* lhs, const ptr& rhs )
{
	return lhs == rhs.get();
}


inline bool operator==( const ptr& lhs, PyObject* rhs )
{
	return lhs.get() == rhs;
}

} // namespace cppy
