/*-----------------------------------------------------------------------------
| Copyright (c) 2013-2017, Nucleic Development Team.
|
| Distributed under the terms of the Modified BSD License.
|
| The full license is in the file COPYING.txt, distributed with this software.
|----------------------------------------------------------------------------*/
#pragma once


namespace kiwi
{

class SharedData
{

public:

    SharedData() : m_refcount( 0 ) {}

    SharedData( const SharedData& other ) : m_refcount( 0 ) {}

    int m_refcount;

private:

    SharedData& operator=( const SharedData& other );
};


template<typename T>
class SharedDataPtr
{

public:

	typedef T Type;

	SharedDataPtr() : m_data( 0 ) {}

	explicit SharedDataPtr( T* data ) : m_data( data )
	{
		incref( m_data );
	}

	~SharedDataPtr()
	{
		decref( m_data );
	}

	T* data()
	{
		return m_data;
	}

	const T* data() const
	{
		return m_data;
	}

	operator T*()
	{
		return m_data;
	}

	operator const T*() const
	{
		return m_data;
	}

	T* operator->()
	{
		return m_data;
	}

	const T* operator->() const
	{
		return m_data;
	}

	T& operator*()
	{
		return *m_data;
	}

	const T& operator*() const
	{
		return *m_data;
	}

	bool operator!() const
	{
		return !m_data;
	}

	bool operator<( const SharedDataPtr<T>& other ) const
	{
		return m_data < other.m_data;
	}

	bool operator==( const SharedDataPtr<T>& other ) const
	{
		return m_data == other.m_data;
	}

	bool operator!=( const SharedDataPtr<T>& other ) const
	{
		return m_data != other.m_data;
	}

	SharedDataPtr( const SharedDataPtr<T>& other ) : m_data( other.m_data )
	{
		incref( m_data );
	}

	SharedDataPtr<T>& operator=( const SharedDataPtr<T>& other )
	{
		if( m_data != other.m_data )
		{
			T* temp = m_data;
			m_data = other.m_data;
			incref( m_data );
			decref( temp );
		}
		return *this;
	}

	SharedDataPtr<T>& operator=( T* other )
	{
		if( m_data != other )
		{
			T* temp = m_data;
			m_data = other;
			incref( m_data );
			decref( temp );
		}
		return *this;
	}

private:

	static void incref( T* data )
	{
		if( data )
			++data->m_refcount;
	}

	static void decref( T* data )
	{
		if( data && --data->m_refcount == 0 )
			delete data;
	}

	T* m_data;
};

} // namespace kiwi
