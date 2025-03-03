/* -----------------------------------------------------------------------------
 * _std_common.i
 *
 * std::helpers for LUA
 * ----------------------------------------------------------------------------- */

%include <std_except.i> // the general exceptions

/*
The basic idea here, is instead of trying to feed SWIG all the
horribly templated STL code, to give it a neatened version.

These %defines cover some of the more common methods
so the class declarations become just a set of %defines

*/

/* #define for basic container features
note: I allow front(), back() & pop_back() to throw exceptions
upon empty containers, rather than coredump
(as we haven't defined the methods, we can use %extend to add with
new features)

*/
%define %STD_CONTAINER_METHODS(CLASS,T)
public:
	CLASS();
	CLASS(const CLASS&);
	unsigned int size() const;
	unsigned int max_size() const;
	bool empty() const;
	void clear();
	%extend {	// the extra stuff which must be checked
		T front()const throw (std::out_of_range){ // only read front & back
			if (self->empty())
				throw std::out_of_range("in "#CLASS"::front()");
			return self->front();
		}
		T back()const throw (std::out_of_range){ // not write to them
			if (self->empty())
				throw std::out_of_range("in "#CLASS"::back()");
			return self->back();
		}
	}
%enddef

/* push/pop for front/back
also note: front & back are read only methods, not used for writing
*/
%define %STD_FRONT_ACCESS_METHODS(CLASS,T)
public:
	void push_front(const T& val);
	%extend {	// must check this
		void pop_front() throw (std::out_of_range){
			if (self->empty())
				throw std::out_of_range("in "#CLASS"::pop_front()");
			self->pop_back();
		}
	}
%enddef

%define %STD_BACK_ACCESS_METHODS(CLASS,T)
public:
	void push_back(const T& val);
	%extend {	// must check this
		void pop_back() throw (std::out_of_range){
			if (self->empty())
				throw std::out_of_range("in "#CLASS"::pop_back()");
			self->pop_back();
		}
	}
%enddef

/*
Random access methods
*/
%define %STD_RANDOM_ACCESS_METHODS(CLASS,T)
	%extend // this is a extra bit of SWIG code
	{
		// [] is replaced by __getitem__ & __setitem__
		// simply throws a string, which causes a lua error
		T __getitem__(unsigned int idx) throw (std::out_of_range){
			if (idx>=self->size())
				throw std::out_of_range("in "#CLASS"::__getitem__()");
			return (*self)[idx];
		}
		void __setitem__(unsigned int idx,const T& val) throw (std::out_of_range){
			if (idx>=self->size())
				throw std::out_of_range("in "#CLASS"::__setitem__()");
			(*self)[idx]=val;
		}
	};
%enddef
