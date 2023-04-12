

%define ADD_EQUALS_AND_HASH_CODE_METHODS(CPPTYPE)

%proxycode %{

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

%enddef

%define ADD_EQUALS_WITH_IMPL_AND_HASH_CODE_METHODS(CPPTYPE)

%extend {
    bool equalsImpl(const CPPTYPE& rhs) const {
        return *self == rhs;
    }
}

ADD_EQUALS_AND_HASH_CODE_METHODS(CPPTYPE)

%enddef

// Useful for wrapper for classes that can be put info TIntrusivePtr
%define ADD_RELEASE_MEM()

%proxycode %{
    void releaseMem() {
        this.swigCMemOwn = false;
    }
%}

%enddef
