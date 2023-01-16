#pragma once

#include <contrib/libs/libunwind/include/unwind.h>

#define DECLARE_PERSONALITY_FUNCTION(name) \
	_Unwind_Reason_Code name(int version,\
			_Unwind_Action actions,\
			uint64_t exceptionClass,\
			struct _Unwind_Exception *exceptionObject,\
			struct _Unwind_Context *context);
#define BEGIN_PERSONALITY_FUNCTION(name) \
	_Unwind_Reason_Code name(int version,\
			_Unwind_Action actions,\
			uint64_t exceptionClass,\
			struct _Unwind_Exception *exceptionObject,\
			struct _Unwind_Context *context)\
{

#define CALL_PERSONALITY_FUNCTION(name) name(version, actions, exceptionClass, exceptionObject, context)

