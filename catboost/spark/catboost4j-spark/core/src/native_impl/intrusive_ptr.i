%{
#include <util/generic/ptr.h>
%}

%include "java_helpers.i"

template <class T>
class TIntrusivePtr {
public:
    T *operator->() const noexcept {
        return pointee;
    }

    inline T* Get() const noexcept;

    %extend {
        // because Reset accepts only TIntrusivePtr and there's no autoconversion in Java
        // use only for SWIG-wrapped classes and release ownership via releaseMem first.
        inline void Set(T* t) noexcept {
            self->Reset(t);
        }
    }

    %typemap(javaimports) TIntrusivePtr<T> "import java.io.*;"
    %typemap(javainterfaces) TIntrusivePtr<T> "Serializable"

    %proxycode %{
        public $javaclassname($typemap(jstype, T) pointee) {
            this();
            if (pointee != null) {
                pointee.releaseMem();
                this.Set(pointee);
            }
        }

        private void writeObject(ObjectOutputStream out) throws IOException {
            out.writeObject(this.Get());
        }

        private void readObject(ObjectInputStream in) throws IOException, ClassNotFoundException {
            this.swigCPtr = native_implJNI.new_$javaclassname();
            this.swigCMemOwn = true;

            $typemap(jboxtype, T) pointee = ($typemap(jboxtype, T))in.readObject();
            if (pointee != null) {
                pointee.releaseMem();
                this.Set(pointee);
            }
        }

        public boolean equals(Object obj) {
            if (obj instanceof $javaclassname) {
                boolean ptrEqual = ((($javaclassname)obj).swigCPtr == this.swigCPtr);
                if (ptrEqual) {
                    return true;
                } else {
                    return this.equalsImpl((($javaclassname)obj).Get());
                }
            } else {
                return false;
            }
        }
    %}

private:
    mutable T* T_;
};
