/*
 * any.h -- a wrapper capable of holding a value of arbitrary type.
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


#ifndef PIRE_ANY_H
#define PIRE_ANY_H


#include <typeinfo>

#include <contrib/libs/pire/pire/stub/stl.h>

namespace Pire {

class Any {

public:
	Any() = default;

	Any(const Any& any)
	{
		if (any.h)
			h = any.h->Duplicate();
	}

	Any& operator= (Any any)
	{
		any.Swap(*this);
		return *this;
	}

	template <class T>
	Any(const T& t)
		: h(new Holder<T>(t))
	{
	}

	bool Empty() const {
		return !h;
	}
	template <class T>
	bool IsA() const {
		return h && h->IsA(typeid(T));
	}

	template <class T>
	T& As()
	{
		if (h && IsA<T>())
			return *reinterpret_cast<T*>(h->Ptr());
		else
			throw Pire::Error("type mismatch");
	}

	template <class T>
	const T& As() const
	{
		if (h && IsA<T>())
			return *reinterpret_cast<const T*>(h->Ptr());
		else
			throw Pire::Error("type mismatch");
	}

	void Swap(Any& a) noexcept {
		DoSwap(h, a.h);
	}

private:

	struct AbstractHolder {
		virtual ~AbstractHolder() {
		}
		virtual THolder<AbstractHolder> Duplicate() const = 0;
		virtual bool IsA(const std::type_info& id) const = 0;
		virtual void* Ptr() = 0;
		virtual const void* Ptr() const = 0;
	};

	template <class T>
	struct Holder: public AbstractHolder {
		Holder(T t)
			: d(t)
		{
		}
		THolder<AbstractHolder> Duplicate() const {
			return THolder<AbstractHolder>(new Holder<T>(d));
		}
		bool IsA(const std::type_info& id) const {
			return id == typeid(T);
		}
		void* Ptr() {
			return &d;
		}
		const void* Ptr() const {
			return &d;
		}
	private:
		T d;
	};

	THolder<AbstractHolder> h;
};

}

namespace std {
	inline void swap(Pire::Any& a, Pire::Any& b) {
		a.Swap(b);
	}
}

#endif
