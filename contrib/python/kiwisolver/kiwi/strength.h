/*-----------------------------------------------------------------------------
| Copyright (c) 2013-2017, Nucleic Development Team.
|
| Distributed under the terms of the Modified BSD License.
|
| The full license is in the file COPYING.txt, distributed with this software.
|----------------------------------------------------------------------------*/
#pragma once
#include <algorithm>


namespace kiwi
{

namespace strength
{

inline double create( double a, double b, double c, double w = 1.0 )
{
	double result = 0.0;
	result += std::max( 0.0, std::min( 1000.0, a * w ) ) * 1000000.0;
	result += std::max( 0.0, std::min( 1000.0, b * w ) ) * 1000.0;
	result += std::max( 0.0, std::min( 1000.0, c * w ) );
	return result;
}


const double required = create( 1000.0, 1000.0, 1000.0 );

const double strong = create( 1.0, 0.0, 0.0 );

const double medium = create( 0.0, 1.0, 0.0 );

const double weak = create( 0.0, 0.0, 1.0 );


inline double clip( double value )
{
	return std::max( 0.0, std::min( required, value ) );
}

} // namespace strength

} // namespace kiwi
