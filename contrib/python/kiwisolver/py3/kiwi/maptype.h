/*-----------------------------------------------------------------------------
| Copyright (c) 2013-2017, Nucleic Development Team.
|
| Distributed under the terms of the Modified BSD License.
|
| The full license is in the file COPYING.txt, distributed with this software.
|----------------------------------------------------------------------------*/
#pragma once
#include <functional>
#include <map>
#include <memory>
#include <utility>
#include "AssocVector.h"


namespace kiwi
{

namespace impl
{

template<
	typename K,
	typename V,
	typename C = std::less<K>,
	typename A = std::allocator< std::pair<K, V> > >
class MapType
{
public:
	typedef Loki::AssocVector<K, V, C, A> Type;
	//typedef std::map<K, V, C, A> Type;
private:
	MapType();
};

} // namespace impl

} // namespace kiwi
