/*
 * pire.h -- a single include file for end-users
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


#ifndef PIRE_PIRE_H
#define PIRE_PIRE_H

#include <contrib/libs/pire/pire/scanners/multi.h>
#include <contrib/libs/pire/pire/scanners/half_final.h>
#include <contrib/libs/pire/pire/scanners/simple.h>
#include <contrib/libs/pire/pire/scanners/slow.h>
#include <contrib/libs/pire/pire/scanners/pair.h>

#include "re_lexer.h"
#include "fsm.h"
#include "encoding.h"
#include "run.h"

#endif
