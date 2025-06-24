/*
 * easy.cpp -- static variables for Pire Easy facilities.
 *
 * Copyright (c) 2007-2010, Dmitry Prokoptsev <dprokoptsev@gmail.com>,
 *                          Alexander Gololobov <agololobov@gmail.com>
 *
 * This file is part of Pire, the Perl Incompatible
 * Regular Expressions library.
 *
 * Pire is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 * 
 * Pire is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser Public License for more details.
 * You should have received a copy of the GNU Lesser Public License
 * along with Pire.  If not, see <http://www.gnu.org/licenses>.
 */

#include "easy.h"

namespace Pire {

const Option<const Encoding&> UTF8(&Pire::Encodings::Utf8);
const Option<const Encoding&> LATIN1(&Pire::Encodings::Latin1);
	
const Option<Feature::Ptr> I(&Pire::Features::CaseInsensitive);
const Option<Feature::Ptr> ANDNOT(&Pire::Features::AndNotSupport);

}
