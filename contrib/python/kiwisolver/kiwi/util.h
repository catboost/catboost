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

namespace impl
{

inline bool nearZero( double value )
{
	const double eps = 1.0e-8;
	return value < 0.0 ? -value < eps : value < eps;
}

} // namespace impl

} // namespace
