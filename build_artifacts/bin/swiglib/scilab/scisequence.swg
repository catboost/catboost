/*
 *
 * Scilab sequence conversions
 *
 */

#define SWIG_Traits_Sequence_frag(Type) %fragment_name(AsVal_Traits_Sequence, Type)

#define SWIG_AsCheck_Sequence_frag(Type...) %fragment_name(AsCheck_Sequence, Type)
#define SWIG_AsCheck_Sequence_dec(Type...) %symbol_name(AsCheck_Sequence, Type)
#define SWIG_AsGet_Sequence_frag(Type...) %fragment_name(AsGet_Sequence, Type)
#define SWIG_AsGet_Sequence_dec(Type...) %symbol_name(AsGet_Sequence, Type)
#define SWIG_AsSize_Sequence_frag(Type...) %fragment_name(AsSize_Sequence, Type)
#define SWIG_AsSize_Sequence_dec(Type...) %symbol_name(AsSize_Sequence, Type)
#define SWIG_FromCreate_Sequence_frag(Type...) %fragment_name(FromCreate_Sequence, Type)
#define SWIG_FromCreate_Sequence_dec(Type...) %symbol_name(FromCreate_Sequence, Type)
#define SWIG_FromSet_Sequence_frag(Type...) %fragment_name(FromSet_Sequence, Type)
#define SWIG_FromSet_Sequence_dec(Type...) %symbol_name(FromSet_Sequence, Type)

#define SWIG_Traits_SequenceItem_frag(Type) %fragment_name(AsVal_Traits_SequenceItem, Type)
#define SWIG_AsVal_SequenceItem_frag(Type...) %fragment_name(AsVal_SequenceItem, Type)
#define SWIG_AsVal_SequenceItem_dec(Type...) %symbol_name(AsVal_SequenceItem, Type)
#define SWIG_From_SequenceItem_frag(Type...) %fragment_name(From_SequenceItem, Type)
#define SWIG_From_SequenceItem_dec(Type...) %symbol_name(From_SequenceItem, Type)

%include <scisequencepointer.swg>
%include <scisequenceint.swg>
%include <scisequencedouble.swg>
%include <scisequencefloat.swg>
%include <scisequencestring.swg>
%include <scisequencebool.swg>

//
// Sequence conversion
//

%fragment(SWIG_Traits_Sequence_frag(ptr), "header",
  fragment=SWIG_AsCheck_Sequence_frag(ptr),
  fragment=SWIG_AsGet_Sequence_frag(ptr),
  fragment=SWIG_AsSize_Sequence_frag(ptr),
  fragment=SWIG_FromCreate_Sequence_frag(ptr),
  fragment=SWIG_FromSet_Sequence_frag(ptr),
  fragment="StdTraits",
  fragment="<stdint.h>") {

namespace swig {
  // Error returned for sequence containers of default item type
  template <typename T> struct traits_as_sequence {
    static int check(SwigSciObject obj) {
      throw std::invalid_argument("The container data type is not supported."); 
    }
    static int get(SwigSciObject obj, void **sequence) {
      throw std::invalid_argument("The container data type is not supported."); 
    }
    static int size(SwigSciObject obj, int *size) {
      throw std::invalid_argument("The container data type is not supported."); 
    }
  };
  template <typename T> struct traits_from_sequence {
    static int create(int size, void **sequence) {
      throw std::invalid_argument("The container data type is not supported."); 
    }
    static SwigSciObject set(int size, void *sequence) {
      throw std::invalid_argument("The container data type is not supported.");       
    }
  };

  // Support sequence containers of pointers
  template <typename T> struct traits_as_sequence<T*> {
    static int check(SwigSciObject obj) {
      return SWIG_AsCheck_Sequence_dec(ptr)(obj);
    }
    static int get(SwigSciObject obj, void **sequence) {
      return SWIG_AsGet_Sequence_dec(ptr)(obj, (int **)sequence);
    }
    static int size(SwigSciObject obj, int *size) {
      return SWIG_AsSize_Sequence_dec(ptr)(obj, size);
    }
  };
  template <typename T> struct traits_from_sequence<T*> {
    static int create(int size, void **sequence) {
      return SWIG_FromCreate_Sequence_dec(ptr)(size, (uintptr_t **)sequence);
    }
    static SwigSciObject set(int size, void *sequence) {
      return SWIG_FromSet_Sequence_dec(ptr)(size, (uintptr_t *)sequence);
    }
  };
}
}

%define %traits_sequence(CppType, ScilabType)
  %fragment(SWIG_Traits_Sequence_frag(CppType), "header",
    fragment=SWIG_Traits_Sequence_frag(ptr),
    fragment=SWIG_AsCheck_Sequence_frag(CppType),
    fragment=SWIG_AsGet_Sequence_frag(CppType),
    fragment=SWIG_AsSize_Sequence_frag(CppType),
    fragment=SWIG_FromCreate_Sequence_frag(CppType),
    fragment=SWIG_FromSet_Sequence_frag(CppType)) {

namespace swig {
  template <> struct traits_as_sequence<CppType > {
    static int check(SwigSciObject obj) {
      return SWIG_AsCheck_Sequence_dec(CppType)(obj);
    }
    static int get(SwigSciObject obj, void **sequence) {
      return SWIG_AsGet_Sequence_dec(CppType)(obj, (ScilabType **)sequence);
    }
    static int size(SwigSciObject obj, int *size) {
      return SWIG_AsSize_Sequence_dec(CppType)(obj, size);
    }
  };
  template <> struct traits_from_sequence<CppType > {
    static int create(int size, void **sequence) {
      return SWIG_FromCreate_Sequence_dec(CppType)(size, (ScilabType **)sequence);
    }
    static SwigSciObject set(int size, void *sequence) {
      return SWIG_FromSet_Sequence_dec(CppType)(size, (ScilabType *)sequence);
    }
  };
}
}
%enddef


//
// Sequence item conversion
//

%fragment(SWIG_Traits_SequenceItem_frag(ptr), "header",
  fragment=SWIG_AsVal_SequenceItem_frag(ptr),
  fragment=SWIG_From_SequenceItem_frag(ptr),
  fragment="StdTraits",
  fragment="<stdint.h>") {

namespace swig {
  // Error returned for sequence containers of default item type
  template <typename T> struct traits_asval_sequenceitem {
    static T asval(SwigSciObject obj, void *pSequence, int iItemIndex) {
      throw std::invalid_argument("The container data type is not supported."); 
    }
  };
  template <typename T> struct traits_from_sequenceitem {
    static int from(void *pSequence, int iItemIndex, T itemValue) {
      throw std::invalid_argument("The container data type is not supported."); 
    }
  };

  // Support sequence containers of pointers
  template <typename T> struct traits_asval_sequenceitem<T*> {
    static T* asval(SwigSciObject obj, void *pSequence, int iItemIndex) {
      return static_cast<T*>(SWIG_AsVal_SequenceItem_dec(ptr)(obj, (int *)pSequence, iItemIndex));
    }
  };
  template <typename T> struct traits_from_sequenceitem<T*> {
    static int from(void *pSequence, int iItemIndex, T *itemValue) {
      return SWIG_From_SequenceItem_dec(ptr)((uintptr_t *)pSequence, iItemIndex, (uintptr_t) itemValue);
     }
   };
}
}

%define %traits_sequenceitem(CppType, ScilabType)
  %fragment(SWIG_Traits_SequenceItem_frag(CppType), "header",
    fragment=SWIG_Traits_SequenceItem_frag(ptr),
    fragment=SWIG_AsVal_SequenceItem_frag(CppType),
    fragment=SWIG_From_SequenceItem_frag(CppType)) {

namespace swig {
  template <> struct traits_asval_sequenceitem<CppType > {
    static CppType asval(SwigSciObject obj, void *pSequence, int iItemIndex) {
      return SWIG_AsVal_SequenceItem_dec(CppType)(obj, (ScilabType *)pSequence, iItemIndex);
    }
  };
  template <> struct traits_from_sequenceitem<CppType > {
    static int from(void *pSequence, int iItemIndex, CppType itemValue) {
      return SWIG_From_SequenceItem_dec(CppType)((ScilabType *)pSequence, iItemIndex, itemValue);
    }
  };
}
}
%enddef

%define %add_traits_sequence(CppType, ScilabType)
  %traits_sequence(CppType, ScilabType);
  %fragment(SWIG_Traits_Sequence_frag(CppType));
  %traits_sequenceitem(CppType, ScilabType);
  %fragment(SWIG_Traits_SequenceItem_frag(CppType));
%enddef

%add_traits_sequence(int, int);
%add_traits_sequence(double, double);
%add_traits_sequence(float, float);
%add_traits_sequence(std::string, char*);
%add_traits_sequence(bool, int);

