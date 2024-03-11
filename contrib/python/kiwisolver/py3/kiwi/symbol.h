/*-----------------------------------------------------------------------------
| Copyright (c) 2013-2017, Nucleic Development Team.
|
| Distributed under the terms of the Modified BSD License.
|
| The full license is in the file LICENSE, distributed with this software.
|----------------------------------------------------------------------------*/
#pragma once


namespace kiwi
{

namespace impl
{

class Symbol
{

public:

	using Id = unsigned long long;

	enum Type
	{
		Invalid,
		External,
		Slack,
		Error,
		Dummy
	};

	Symbol() : m_id( 0 ), m_type( Invalid ) {}

	Symbol( Type type, Id id ) : m_id( id ), m_type( type ) {}

	~Symbol() = default;

	Id id() const
	{
		return m_id;
	}

	Type type() const
	{
		return m_type;
	}

private:

	Id m_id;
	Type m_type;

	friend bool operator<( const Symbol& lhs, const Symbol& rhs )
	{
		return lhs.m_id < rhs.m_id;
	}

	friend bool operator==( const Symbol& lhs, const Symbol& rhs )
	{
		return lhs.m_id == rhs.m_id;
	}

};

} // namespace impl

} // namespace kiwi
