%{
#include <util/generic/maybe.h>
%}

%include "java_helpers.i"

template <class T>
class TMaybe {
public:
    TMaybe();
    TMaybe(const T& arg);

    constexpr bool Defined() const noexcept;
    constexpr const T& GetRef() const&;


    %typemap(javaimports) TMaybe<T> "import java.io.*;"
    %typemap(javainterfaces) TMaybe<T> "Serializable"

    %extend {
        // need this because it is impossible to call non-default constructor for 'this' from deserialization
        void MoveInto(TMaybe<T>* rhs) {
            (*self) = std::move(*rhs);
        }
    }

    %proxycode %{
        private void writeObject(ObjectOutputStream out) throws IOException {
            boolean isDefined = this.Defined();
            out.writeBoolean(isDefined);
            if (isDefined) {
                out.writeUnshared(this.GetRef());
            }
        }

        private void readObject(ObjectInputStream in) throws IOException, ClassNotFoundException {
            this.swigCPtr = native_implJNI.new_$javaclassname__SWIG_0();
            this.swigCMemOwn = true;

            boolean isDefined = in.readBoolean();
            if (isDefined) {
                this.MoveInto(new $javaclassname(($typemap(jboxtype, T))in.readUnshared()));
            }
        }
    %}

    ADD_EQUALS_WITH_IMPL_AND_HASH_CODE_METHODS(TMaybe<T>)
};

