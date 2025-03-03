/* -----------------------------------------------------------------------------
 * std_list.i
 *
 * SWIG typemaps for std::list types
 * ----------------------------------------------------------------------------- */

%include <std_common.i>

%module std_list
%{
#include <list>
#include <stdexcept>
%}


namespace std {
    template<class T> class list
    {
    public:
	typedef size_t size_type;
	typedef ptrdiff_t difference_type;
	typedef T value_type;
	typedef value_type* pointer;
	typedef const value_type* const_pointer;
	typedef value_type& reference;
	typedef const value_type& const_reference;
	typedef T &iterator;
	typedef const T& const_iterator; 

	list();
	list(unsigned int size, const T& value = T());
	list(const list& other);

	void assign(unsigned int n, const T& value);
	void swap(list<T> &x);

	const_reference front();
	const_reference back();
	const_iterator begin();
	const_iterator end();

	void resize(unsigned int n, T c = T());
	bool empty() const;

	void push_front(const T& x);
	void push_back(const T& x);

	void pop_front();
	void pop_back();
	void clear();
	unsigned int size() const;
	unsigned int max_size() const;
	void resize(unsigned int n, const T& value);

	void remove(const T& value);
	void unique();
	void reverse();
	void sort();

	%extend 
	    {
		const_reference __getitem__(int i) throw (std::out_of_range) 
		    {
			std::list<T>::iterator first = self->begin(); 
			int size = int(self->size());
			if (i<0) i += size;
			if (i>=0 && i<size)
			{
			    for (int k=0;k<i;k++)
			    {
				first++;
			    }
			    return *first;
			}
			else throw std::out_of_range("list index out of range");
		    }
		void __setitem__(int i, const T& x) throw (std::out_of_range)
		    {
			std::list<T>::iterator first = self->begin(); 
			int size = int(self->size());
			if (i<0) i += size;
			if (i>=0 && i<size)
			{
			    for (int k=0;k<i;k++)
			    {
				first++;
			    }
			    *first = x;
			}
			else throw std::out_of_range("list index out of range");
		    }
		void __delitem__(int i) throw (std::out_of_range)
		    {
			std::list<T>::iterator first = self->begin(); 
			int size = int(self->size());
			if (i<0) i += size;
			if (i>=0 && i<size)
			{
			    for (int k=0;k<i;k++)
			    {
				first++;
			    }
			    self->erase(first);
			}
			else throw std::out_of_range("list index out of range");
		    }	     
		std::list<T> __getslice__(int i,int j) 
		    {
			std::list<T>::iterator first = self->begin();
			std::list<T>::iterator end = self->end();

			int size = int(self->size());
			if (i<0) i += size;
			if (j<0) j += size;
			if (i<0) i = 0;
			if (j>size) j = size;
			if (i>=j) i=j;
			if (i>=0 && i<size && j>=0)
			{
			    for (int k=0;k<i;k++)
			    {
				first++;
			    }
			    for (int m=0;m<j;m++)
			    {
				end++;
			    }
			    std::list<T> tmp(j-i);
			    if (j>i) std::copy(first,end,tmp.begin());
			    return tmp;
			}
			else throw std::out_of_range("list index out of range");
		    }
		void __delslice__(int i,int j) 
		    {
			std::list<T>::iterator first = self->begin();
			std::list<T>::iterator end = self->end();

			int size = int(self->size());
			if (i<0) i += size;
			if (j<0) j += size;
			if (i<0) i = 0;
			if (j>size) j = size;
	
			for (int k=0;k<i;k++)
			{
			    first++;
			}
			for (int m=0;m<=j;m++)
			{
			    end++;
			}		   
			self->erase(first,end);		
		    }
		void __setslice__(int i,int j, const std::list<T>& v) 
		    {
			std::list<T>::iterator first = self->begin();
			std::list<T>::iterator end = self->end();

			int size = int(self->size());
			if (i<0) i += size;
			if (j<0) j += size;
			if (i<0) i = 0;
			if (j>size) j = size;
		
			for (int k=0;k<i;k++)
			{
			    first++;
			}
			for (int m=0;m<=j;m++)
			{
			    end++;
			}
			if (int(v.size()) == j-i) 
			{
			    std::copy(v.begin(),v.end(),first);
			}
			else {
			    self->erase(first,end);
			    if (i+1 <= int(self->size())) 
			    {
				first = self->begin();
				for (int k=0;k<i;k++)
				{
				    first++;
				}
				self->insert(first,v.begin(),v.end());
			    }
			    else self->insert(self->end(),v.begin(),v.end());
			}
			   	
		    }
		unsigned int __len__() 
		    {
			return self->size();
		    }	
		bool __nonzero__()
		    {
			return !(self->empty());
		    }
		void append(const T& x)
		    {
			self->push_back(x);
		    }
		void pop()
		    {
			self->pop_back();
		    }
	    }
    };
}






