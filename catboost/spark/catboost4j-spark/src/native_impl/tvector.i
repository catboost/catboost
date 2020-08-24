/*
 * TODO(akhropov) Cannot reuse SWIG's std::vector implementation because SWIG's std::vector for Java
 *   does not contain allocator argument which is specified in TVector's declaration
 */

%{
#include <util/generic/vector.h>
#include <stdexcept>
%}

%include <typemaps.i>
%include <std_except.i>

template <class T>
class TVector {

    %typemap(javabase) TVector<T> "java.util.AbstractList<$typemap(jboxtype, T)>"
    %typemap(javainterfaces) TVector<T> "java.util.RandomAccess"

public:
    void reserve(size_t new_cap);

    %extend {
        bool equalsImpl(const TVector<T>& rhs) const  throw (std::exception) {
            return *self == rhs;
        }

        const T& getImpl(jint index) const throw (std::out_of_range) {
            if ((index < 0) || (index >= (jint)self->size())) {
                throw std::out_of_range("TVector index is out of range");
            }
            return (*self)[index];
        }

        T setImpl(jint index, const T& element) throw (std::out_of_range) {
            if ((index < 0) || (index >= (jint)self->size())) {
                throw std::out_of_range("TVector index is out of range");
            }
            T oldValue = (*self)[index];
            (*self)[index] = element;
            return oldValue;
        }

        jint sizeImpl() const throw (std::out_of_range) {
            size_t size = self->size();
            if (size > Max<jint>()) {
                throw std::out_of_range("TVector size cannot be represented by JVM's int type");
            }
            return (jint)size;
        }

        void addImpl(const T& element) throw (std::exception) {
            self->push_back(element);
        }

        void addImpl(jint index, const T& element) throw (std::out_of_range) {
            if ((index < 0) || (index > (jint)self->size())) {
                throw std::out_of_range("TVector index is out of range");
            }
            self->insert(self->begin() + index, element);
        }

        T removeImpl(jint index) throw (std::out_of_range) {
            if ((index < 0) || (index >= (jint)self->size())) {
                throw std::out_of_range("TVector index is out of range");
            }
            T oldValue = (*self)[index];
            self->erase(self->begin() + index);
            return oldValue;
        }
    }
    
    %proxycode %{
        public $javaclassname($typemap(jstype, T)[] elements) {
            this();
            reserve(elements.length);
            for ($typemap(jstype, T) element : elements) {
                addImpl(element);
            }
        }

        public $javaclassname(Iterable<$typemap(jboxtype, T)> elements) {
            this();
            for ($typemap(jstype, T) element : elements) {
                addImpl(element);
            }
        }

        public $typemap(jboxtype, T) get(int index) {
            return getImpl(index);
        }

        public $typemap(jboxtype, T) set(int index, $typemap(jboxtype, T) element) {
            return setImpl(index, element);
        }

        public int size() {
            return sizeImpl();
        }
        
        public boolean add($typemap(jboxtype, T) element) {
            modCount++;
            addImpl(element);
            return true;
        }
        
        public void add(int index, $typemap(jboxtype, T) element) {
            modCount++;
            addImpl(index, element);
        }

        public $typemap(jboxtype, T) remove(int index) {
            modCount++;
            return removeImpl(index);
        }

        public boolean equals(Object obj) {
            if (obj instanceof $javaclassname) {
                boolean ptrEqual = ((($javaclassname)obj).swigCPtr == this.swigCPtr);
                if (ptrEqual) {
                    return true;
                } else {
                    return this.equalsImpl(($javaclassname)obj);
                }
            } else {
                return false;
            }
        }
        
        public int hashCode() {
            return (int)this.swigCPtr;
        }
    %}
};
