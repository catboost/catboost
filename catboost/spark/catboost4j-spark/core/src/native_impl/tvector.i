/*
 * TODO(akhropov) Cannot reuse SWIG's std::vector implementation because SWIG's std::vector for Java
 *   does not contain allocator argument which is specified in TVector's declaration
 */

%{
#include <util/generic/vector.h>
#include <util/generic/ylimits.h>
#include <stdexcept>
%}

%include <typemaps.i>
%include <std_except.i>

%include "java_helpers.i"
%include "primitive_arrays.i"


template <class T>
class TVector {

    %typemap(javaimports) TVector<T> "import java.io.*;"
    %typemap(javabase) TVector<T> "java.util.AbstractList<$typemap(jboxtype, T)>"
    %typemap(javainterfaces) TVector<T> "java.util.RandomAccess,Serializable"

public:
    void reserve(size_t new_cap);
    
    %extend {
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
        
        // Generic serialization implementation - not very fast
        
        private void writeObject(ObjectOutputStream out) throws IOException {
            int length = this.size();
            out.writeInt(length);
            for (int i = 0; i < length; ++i) {
                out.writeObject(this.get(i));
            }
        }
    
        private void readObject(ObjectInputStream in) throws IOException, ClassNotFoundException {
            this.swigCPtr = native_implJNI.new_$javaclassname();
            this.swigCMemOwn = true;
        
            int length = in.readInt();
            this.reserve(length);
            for (int i = 0; i < length; ++i) {
               this.add(($typemap(jboxtype, T))in.readObject());
            }
        }
    %}
    
    ADD_EQUALS_WITH_IMPL_AND_HASH_CODE_METHODS(TVector<T>)
};


%define EXTEND_FOR_PRIMITIVE_TYPE(CPPTYPE, JNITYPE, JAVATYPE)
    %template(TVector_##CPPTYPE) TVector<CPPTYPE>;

    %native (toPrimitiveArrayImpl_##CPPTYPE) JNITYPE##Array toPrimitiveArrayImpl_##CPPTYPE(const TVector<CPPTYPE>& v);
    %{
    extern "C" {
        JNIEXPORT JNITYPE##Array JNICALL 
        Java_ru_yandex_catboost_spark_catboost4j_1spark_core_src_native_1impl_native_1implJNI_toPrimitiveArrayImpl_1##CPPTYPE(
            JNIEnv* jenv,
            jclass /*cls*/,
            jlong jarg1,
            jobject /*jarg1_*/
        ) {
            TVector<CPPTYPE>* self = *(TVector<CPPTYPE>**)&jarg1;  
            JNITYPE##Array result;
            try {
                result = SWIG_PrimitiveArrayCppToJava(jenv, self->data(), self->size());
            } catch (std::exception& e) {
                SWIG_JavaThrowException(jenv, SWIG_JavaRuntimeException, e.what());
            } catch (...) {
                SWIG_JavaThrowException(jenv, SWIG_JavaRuntimeException, "Unknown C++ exception thrown");
            }
            return result;
        } 
    }  
        
    %}

    %extend TVector<CPPTYPE> {
        %proxycode %{
            public JAVATYPE[] toPrimitiveArray() {
                return $imclassname.toPrimitiveArrayImpl_##CPPTYPE(swigCPtr, this);
            }
        %}
    }
    
%enddef


EXTEND_FOR_PRIMITIVE_TYPE(i8, jbyte, byte);
EXTEND_FOR_PRIMITIVE_TYPEL(ui16, jchar, char);
EXTEND_FOR_PRIMITIVE_TYPE(i16, jshort, short);
EXTEND_FOR_PRIMITIVE_TYPE(i32, jint, int);
//EXTEND_FOR_PRIMITIVE_TYPE(i64, jlong, long);
EXTEND_FOR_PRIMITIVE_TYPE(float, jfloat, float);
EXTEND_FOR_PRIMITIVE_TYPE(double, jdouble, double);

