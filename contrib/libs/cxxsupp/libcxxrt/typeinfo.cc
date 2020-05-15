/* 
 * Copyright 2010-2012 PathScale, Inc. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 * 
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS ``AS
 * IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
 * THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
 * OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
 * WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
 * OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
 * ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include "typeinfo.h"
#include <string.h>
#include <stdlib.h>
#include <stdio.h>

using std::type_info;

type_info::~type_info() {}

bool type_info::operator==(const type_info &other) const
{
	return __type_name == other.__type_name;
}
bool type_info::operator!=(const type_info &other) const
{
	return __type_name != other.__type_name;
}
bool type_info::before(const type_info &other) const
{
	return __type_name < other.__type_name;
}
const char* type_info::name() const
{
	return __type_name;
}
type_info::type_info (const type_info& rhs)
{
	__type_name = rhs.__type_name;
}
type_info& type_info::operator= (const type_info& rhs) 
{
	return *new type_info(rhs);
}

ABI_NAMESPACE::__fundamental_type_info::~__fundamental_type_info() {}
ABI_NAMESPACE::__array_type_info::~__array_type_info() {}
ABI_NAMESPACE::__function_type_info::~__function_type_info() {}
ABI_NAMESPACE::__enum_type_info::~__enum_type_info() {}
ABI_NAMESPACE::__class_type_info::~__class_type_info() {}
ABI_NAMESPACE::__si_class_type_info::~__si_class_type_info() {}
ABI_NAMESPACE::__vmi_class_type_info::~__vmi_class_type_info() {}
ABI_NAMESPACE::__pbase_type_info::~__pbase_type_info() {}
ABI_NAMESPACE::__pointer_type_info::~__pointer_type_info() {}
ABI_NAMESPACE::__pointer_to_member_type_info::~__pointer_to_member_type_info() {}
