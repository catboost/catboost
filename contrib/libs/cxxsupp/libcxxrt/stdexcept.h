/* 
 * Copyright 2010-2011 PathScale, Inc. All rights reserved.
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

/**
 * stdexcept.h - provides a stub version of <stdexcept>, which defines enough
 * of the exceptions for the runtime to use.  
 */

namespace std
{

	class exception
	{
	public:
		exception() noexcept;
		exception(const exception&) noexcept;
		exception& operator=(const exception&) noexcept;
		virtual ~exception();
		virtual const char* what() const noexcept;
	};


	/**
	 * Bad allocation exception.  Thrown by ::operator new() if it fails.
	 */
	class bad_alloc: public exception
	{
	public:
		bad_alloc() noexcept;
		bad_alloc(const bad_alloc&) noexcept;
		bad_alloc& operator=(const bad_alloc&) noexcept;
		~bad_alloc();
		virtual const char* what() const noexcept;
	};

	/**
	 * Bad cast exception.  Thrown by the __cxa_bad_cast() helper function.
	 */
	class bad_cast: public exception {
	public:
		bad_cast() noexcept;
		bad_cast(const bad_cast&) noexcept;
		bad_cast& operator=(const bad_cast&) noexcept;
		virtual ~bad_cast();
		virtual const char* what() const noexcept;
	};

	/**
	 * Bad typeidexception.  Thrown by the __cxa_bad_typeid() helper function.
	 */
	class bad_typeid: public exception
	{
	public:
		bad_typeid() noexcept;
		bad_typeid(const bad_typeid &__rhs) noexcept;
		virtual ~bad_typeid();
		bad_typeid& operator=(const bad_typeid &__rhs) noexcept;
		virtual const char* what() const noexcept;
	};

	class bad_array_new_length: public bad_alloc
	{
	public:
		bad_array_new_length() noexcept;
		bad_array_new_length(const bad_array_new_length&) noexcept;
		bad_array_new_length& operator=(const bad_array_new_length&) noexcept;
		virtual ~bad_array_new_length();
		virtual const char *what() const noexcept;
	};


} // namespace std

