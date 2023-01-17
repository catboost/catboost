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
    void yresize(size_t new_size);

    void reserve(size_t new_cap);

    %extend {
        const T& getImpl(jint index) const {
            if ((index < 0) || (index >= (jint)self->size())) {
                throw std::out_of_range("TVector index is out of range");
            }
            return (*self)[index];
        }

        T setImpl(jint index, const T& element) {
            if ((index < 0) || (index >= (jint)self->size())) {
                throw std::out_of_range("TVector index is out of range");
            }
            T oldValue = (*self)[index];
            (*self)[index] = element;
            return oldValue;
        }

        jint sizeImpl() const {
            size_t size = self->size();
            if (size > Max<jint>()) {
                throw std::out_of_range("TVector size cannot be represented by JVM's int type");
            }
            return (jint)size;
        }

        void addImpl(const T& element) {
            self->push_back(element);
        }

        void addImpl(jint index, const T& element) {
            if ((index < 0) || (index > (jint)self->size())) {
                throw std::out_of_range("TVector index is out of range");
            }
            self->insert(self->begin() + index, element);
        }

        T removeImpl(jint index) {
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

%define DECLARE_TVECTOR(NAME, CPPTYPE)

    %catches(std::exception) NAME::yresize(size_t new_size);
    %catches(std::exception) NAME::reserve(size_t new_cap);
    %catches(std::out_of_range) NAME::getImpl(jint index);
    %catches(std::out_of_range) NAME::sizeImpl() const;
    %catches(std::exception) NAME::addImpl(const T& element);
    %catches(std::out_of_range) NAME::addImpl(jint index, const T& element);
    %catches(std::out_of_range) NAME::removeImpl(jint index);
    %catches(std::exception) NAME::equalsImpl(const NAME& rhs);

    %template(NAME) TVector<CPPTYPE>;

%enddef


%define EXTEND_FOR_PRIMITIVE_TYPE(CPPTYPE, JNITYPE, JAVATYPE)
    DECLARE_TVECTOR(TVector_##CPPTYPE, CPPTYPE);

    %native (toPrimitiveArrayImpl_##CPPTYPE) JNITYPE##Array toPrimitiveArrayImpl_##CPPTYPE(const TVector<CPPTYPE>& v);
    %native (asDirectByteBufferImpl_##CPPTYPE) jobject asDirectByteBufferImpl_##CPPTYPE(const TVector<CPPTYPE>& v);
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

        JNIEXPORT jobject JNICALL
        Java_ru_yandex_catboost_spark_catboost4j_1spark_core_src_native_1impl_native_1implJNI_asDirectByteBufferImpl_1##CPPTYPE(
            JNIEnv* jenv,
            jclass /*cls*/,
            jlong jarg1,
            jobject /*jarg1_*/
        ) {
            TVector<CPPTYPE>* self = *(TVector<CPPTYPE>**)&jarg1;
            jobject result = nullptr;
            size_t sizeInBytes = self->size() * sizeof(CPPTYPE);
            if (sizeInBytes > Max<jlong>()) {
                SWIG_JavaThrowException(
                    jenv,
                    SWIG_JavaRuntimeException,
                    "Size of vector is too big for java.nio.ByteBuffer"
                );
            } else {
                result = jenv->NewDirectByteBuffer(self->data(), (jlong)sizeInBytes);
                if (!result && !jenv->ExceptionCheck()) {
                    SWIG_JavaThrowException(
                        jenv,
                        SWIG_JavaRuntimeException,
                        "JNI access to direct buffers is not supported by JVM."
                    );
                }
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

            // valid only until next reallocation of TVector, so, use with caution.
            public java.nio.ByteBuffer asDirectByteBuffer() {
                return (java.nio.ByteBuffer)$imclassname.asDirectByteBufferImpl_##CPPTYPE(swigCPtr, this);
            }
        %}
    }

%enddef


EXTEND_FOR_PRIMITIVE_TYPE(i8, jbyte, byte);
EXTEND_FOR_PRIMITIVE_TYPEL(ui16, jchar, char);
EXTEND_FOR_PRIMITIVE_TYPE(i16, jshort, short);
EXTEND_FOR_PRIMITIVE_TYPE(i32, jint, int);
EXTEND_FOR_PRIMITIVE_TYPE(i64, jlong, long);
EXTEND_FOR_PRIMITIVE_TYPE(float, jfloat, float);
EXTEND_FOR_PRIMITIVE_TYPE(double, jdouble, double);

