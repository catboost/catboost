// R specific swig components
/*
  Vectors
*/

%fragment("StdVectorTraits","header",fragment="StdSequenceTraits")
%{
  namespace swig {
    // vectors of doubles
    template <>
      struct traits_from_ptr<std::vector<double> > {
      static SEXP from (std::vector<double > *val, int owner = 0) {
        SEXP result;
        PROTECT(result = Rf_allocVector(REALSXP, val->size()));
        for (unsigned pos = 0; pos < val->size(); pos++)
          {
            NUMERIC_POINTER(result)[pos] = ((*val)[pos]);
          }
        UNPROTECT(1);
        return(result);
      }
    };
    // vectors of floats
    template <>
      struct traits_from_ptr<std::vector<float> > {
      static SEXP from (std::vector<float > *val, int owner = 0) {
        SEXP result;
        PROTECT(result = Rf_allocVector(REALSXP, val->size()));
        for (unsigned pos = 0; pos < val->size(); pos++)
          {
            NUMERIC_POINTER(result)[pos] = ((*val)[pos]);
          }
        UNPROTECT(1);
        return(result);
      }
    };
    // vectors of unsigned 8bit int
    template <>
      struct traits_from_ptr<std::vector<unsigned char> > {
      static SEXP from (std::vector<unsigned char > *val, int owner = 0) {
        SEXP result;
        PROTECT(result = Rf_allocVector(INTSXP, val->size()));
        for (unsigned pos = 0; pos < val->size(); pos++)
          {
            INTEGER_POINTER(result)[pos] = ((*val)[pos]);
          }
        UNPROTECT(1);
        return(result);
      }
    };
    // vectors of 8bit int
    template <>
      struct traits_from_ptr<std::vector<signed char> > {
      static SEXP from (std::vector<signed char > *val, int owner = 0) {
        SEXP result;
        PROTECT(result = Rf_allocVector(INTSXP, val->size()));
        for (unsigned pos = 0; pos < val->size(); pos++)
          {
            INTEGER_POINTER(result)[pos] = ((*val)[pos]);
          }
        UNPROTECT(1);
        return(result);
      }
    };

    // vectors of unsigned 16bit int
    template <>
      struct traits_from_ptr<std::vector<unsigned short int> > {
      static SEXP from (std::vector<unsigned short int > *val, int owner = 0) {
        SEXP result;
        PROTECT(result = Rf_allocVector(INTSXP, val->size()));
        for (unsigned pos = 0; pos < val->size(); pos++)
          {
            INTEGER_POINTER(result)[pos] = ((*val)[pos]);
          }
        UNPROTECT(1);
        return(result);
      }
    };
    // vectors of 16bit int
    template <>
      struct traits_from_ptr<std::vector<short int> > {
      static SEXP from (std::vector<short int > *val, int owner = 0) {
        SEXP result;
        PROTECT(result = Rf_allocVector(INTSXP, val->size()));
        for (unsigned pos = 0; pos < val->size(); pos++)
          {
            INTEGER_POINTER(result)[pos] = ((*val)[pos]);
          }
        UNPROTECT(1);
        return(result);
      }
    };

   // vectors of 32 bit unsigned int
    template <>
      struct traits_from_ptr<std::vector<unsigned int> > {
      static SEXP from (std::vector<unsigned int> *val, int owner = 0) {
        SEXP result;
        PROTECT(result = Rf_allocVector(INTSXP, val->size()));
        for (unsigned pos = 0; pos < val->size(); pos++)
          {
            INTEGER_POINTER(result)[pos] = ((*val)[pos]);
          }
        UNPROTECT(1);
        return(result);
      }
    };

    // vectors of 32bit int
    template <>
      struct traits_from_ptr<std::vector<int> > {
      static SEXP from (std::vector<int > *val, int owner = 0) {
        SEXP result;
        PROTECT(result = Rf_allocVector(INTSXP, val->size()));
        for (unsigned pos = 0; pos < val->size(); pos++)
          {
            INTEGER_POINTER(result)[pos] = ((*val)[pos]);
          }
        UNPROTECT(1);
        return(result);
      }
    };

   // vectors of 64 bit unsigned int
#if defined(SWIGWORDSIZE64)
    template <>
      struct traits_from_ptr<std::vector<unsigned long int> > {
      static SEXP from (std::vector<unsigned long int> *val, int owner = 0) {
        SEXP result;
        PROTECT(result = Rf_allocVector(INTSXP, val->size()));
        for (unsigned pos = 0; pos < val->size(); pos++)
          {
            INTEGER_POINTER(result)[pos] = ((*val)[pos]);
          }
        UNPROTECT(1);
        return(result);
      }
    };
     // vectors of 64 bit int
    template <>
      struct traits_from_ptr<std::vector<long int> > {
      static SEXP from (std::vector<long int> *val, int owner = 0) {
        SEXP result;
        PROTECT(result = Rf_allocVector(INTSXP, val->size()));
        for (unsigned pos = 0; pos < val->size(); pos++)
          {
            INTEGER_POINTER(result)[pos] = ((*val)[pos]);
          }
        UNPROTECT(1);
        return(result);
      }
    };
#else
    template <>
      struct traits_from_ptr<std::vector<unsigned long long int> > {
      static SEXP from (std::vector<unsigned long long int> *val, int owner = 0) {
        SEXP result;
        PROTECT(result = Rf_allocVector(INTSXP, val->size()));
        for (unsigned pos = 0; pos < val->size(); pos++)
          {
            INTEGER_POINTER(result)[pos] = ((*val)[pos]);
          }
        UNPROTECT(1);
        return(result);
      }
    };
     // vectors of 64 bit int
    template <>
      struct traits_from_ptr<std::vector<long long int> > {
      static SEXP from (std::vector<long long int> *val, int owner = 0) {
        SEXP result;
        PROTECT(result = Rf_allocVector(INTSXP, val->size()));
        for (unsigned pos = 0; pos < val->size(); pos++)
          {
            INTEGER_POINTER(result)[pos] = ((*val)[pos]);
          }
        UNPROTECT(1);
        return(result);
      }
    };
#endif
    // vectors of bool
    template <>
      struct traits_from_ptr<std::vector<bool> > {
      static SEXP from (std::vector<bool> *val, int owner = 0) {
        SEXP result;
        PROTECT(result = Rf_allocVector(LGLSXP, val->size()));
        for (unsigned pos = 0; pos < val->size(); pos++)
          {
            LOGICAL_POINTER(result)[pos] = ((*val)[pos]);
          }
        UNPROTECT(1);
        return(result);
      }
    };
    
    // vectors of strings
    template <>
      struct traits_from_ptr<std::vector<std::basic_string<char> > > {
      static SEXP from (std::vector<std::basic_string<char> > *val, int owner = 0) {
        SEXP result;
         PROTECT(result = Rf_allocVector(STRSXP, val->size()));
         for (unsigned pos = 0; pos < val->size(); pos++)
           {
             CHARACTER_POINTER(result)[pos] = Rf_mkChar(((*val)[pos]).c_str());
           }
        UNPROTECT(1);
        return(result);
      }
    };

    // catch all that does everything with vectors
    template <typename T>
      struct traits_from_ptr< std::vector< T > > {
      static SEXP from (std::vector< T > *val, int owner = 0) {
        return SWIG_R_NewPointerObj(val, type_info< std::vector< T >  >(), owner);
      }
    };
    /////////////////////////////////////////////////
    template <>
  struct traits_asptr < std::vector<double> > {
    static int asptr(SEXP obj, std::vector<double> **val) {
      std::vector<double> *p;
      // not sure how to check the size of the SEXP obj is correct
      unsigned int sexpsz = Rf_length(obj);
      p = new std::vector<double>(sexpsz);
      double *S = NUMERIC_POINTER(obj);
      for (unsigned pos = 0; pos < p->size(); pos++)
        {
          (*p)[pos] = static_cast<double>(S[pos]);
        }
      int res = SWIG_OK;
      if (SWIG_IsOK(res)) {
        if (val) *val = p;
      }
      return res;
    }
  };

    template <>
  struct traits_asptr < std::vector<float> > {
    static int asptr(SEXP obj, std::vector<float> **val) {
      std::vector<float> *p;
      // not sure how to check the size of the SEXP obj is correct
      unsigned int sexpsz = Rf_length(obj);
      p = new std::vector<float>(sexpsz);
      double *S = NUMERIC_POINTER(obj);
      for (unsigned pos = 0; pos < p->size(); pos++)
        {
          (*p)[pos] = static_cast<double>(S[pos]);
        }
      int res = SWIG_OK;
      if (SWIG_IsOK(res)) {
        if (val) *val = p;
      }
      return res;
    }
  };

    // 8 bit integer types
    template <>
  struct traits_asptr < std::vector<unsigned char> > {
    static int asptr(SEXP obj, std::vector<unsigned char> **val) {
      std::vector<unsigned char> *p;
      unsigned int sexpsz = Rf_length(obj);
      p = new std::vector<unsigned char>(sexpsz);
      SEXP coerced;
      PROTECT(coerced = Rf_coerceVector(obj, INTSXP));
      int *S = INTEGER_POINTER(coerced);
      for (unsigned pos = 0; pos < p->size(); pos++)
        {
          (*p)[pos] = static_cast<unsigned char>(S[pos]);
        }
      int res = SWIG_OK;
      if (SWIG_IsOK(res)) {
        if (val) *val = p;
      }
      UNPROTECT(1);
      return res;
    }
  };

    template <>
  struct traits_asptr < std::vector<signed char> > {
    static int asptr(SEXP obj, std::vector<signed char> **val) {
      std::vector<signed char> *p;
      // not sure how to check the size of the SEXP obj is correct
      int sexpsz = Rf_length(obj);
      p = new std::vector<signed char>(sexpsz);
      SEXP coerced;
      PROTECT(coerced = Rf_coerceVector(obj, INTSXP));
      int *S = INTEGER_POINTER(coerced);
      for (unsigned pos = 0; pos < p->size(); pos++)
        {
          (*p)[pos] = static_cast<signed char>(S[pos]);
        }
      int res = SWIG_OK;
      if (SWIG_IsOK(res)) {
        if (val) *val = p;
      }
      UNPROTECT(1);
      return res;
    }
  };

   // 16 bit integer types
    template <>
  struct traits_asptr < std::vector<unsigned short int> > {
    static int asptr(SEXP obj, std::vector<unsigned short int> **val) {
      std::vector<unsigned short int> *p;
      unsigned int sexpsz = Rf_length(obj);
      p = new std::vector<unsigned short int>(sexpsz);
      SEXP coerced;
      PROTECT(coerced = Rf_coerceVector(obj, INTSXP));
      int *S = INTEGER_POINTER(coerced);
      for (unsigned pos = 0; pos < p->size(); pos++)
        {
          (*p)[pos] = static_cast<unsigned short int>(S[pos]);
        }
      int res = SWIG_OK;
      if (SWIG_IsOK(res)) {
        if (val) *val = p;
      }
      UNPROTECT(1);
      return res;
    }
  };

    template <>
  struct traits_asptr < std::vector<short int> > {
    static int asptr(SEXP obj, std::vector<short int> **val) {
      std::vector<short int> *p;
      // not sure how to check the size of the SEXP obj is correct
      int sexpsz = Rf_length(obj);
      p = new std::vector<short int>(sexpsz);
      SEXP coerced;
      PROTECT(coerced = Rf_coerceVector(obj, INTSXP));
      int *S = INTEGER_POINTER(coerced);
      for (unsigned pos = 0; pos < p->size(); pos++)
        {
          (*p)[pos] = static_cast<short int>(S[pos]);
        }
      int res = SWIG_OK;
      if (SWIG_IsOK(res)) {
        if (val) *val = p;
      }
      UNPROTECT(1);
      return res;
    }
  };
    // 32 bit integer types
    template <>
  struct traits_asptr < std::vector<unsigned int> > {
    static int asptr(SEXP obj, std::vector<unsigned int> **val) {
      std::vector<unsigned int> *p;
      unsigned int sexpsz = Rf_length(obj);
      p = new std::vector<unsigned int>(sexpsz);
      SEXP coerced;
      PROTECT(coerced = Rf_coerceVector(obj, INTSXP));
      int *S = INTEGER_POINTER(coerced);
      for (unsigned pos = 0; pos < p->size(); pos++)
        {
          (*p)[pos] = static_cast<unsigned int>(S[pos]);
        }
      int res = SWIG_OK;
      if (SWIG_IsOK(res)) {
        if (val) *val = p;
      }
      UNPROTECT(1);
      return res;
    }
  };

    template <>
  struct traits_asptr < std::vector<int> > {
    static int asptr(SEXP obj, std::vector<int> **val) {
      std::vector<int> *p;
      // not sure how to check the size of the SEXP obj is correct
      int sexpsz = Rf_length(obj);
      p = new std::vector<int>(sexpsz);
      SEXP coerced;
      PROTECT(coerced = Rf_coerceVector(obj, INTSXP));
      int *S = INTEGER_POINTER(coerced);
      for (unsigned pos = 0; pos < p->size(); pos++)
        {
          (*p)[pos] = static_cast<int>(S[pos]);
        }
      int res = SWIG_OK;
      if (SWIG_IsOK(res)) {
        if (val) *val = p;
      }
      UNPROTECT(1);
      return res;
    }
  };

#if defined(SWIGWORDSIZE64)
    // 64 bit integer types
    template <>
  struct traits_asptr < std::vector<unsigned long int> > {
    static int asptr(SEXP obj, std::vector<unsigned long int> **val) {
      std::vector<unsigned long int> *p;
      unsigned int sexpsz = Rf_length(obj);
      p = new std::vector<unsigned long int>(sexpsz);
      SEXP coerced;
      PROTECT(coerced = Rf_coerceVector(obj, INTSXP));
      int *S = INTEGER_POINTER(coerced);
      for (unsigned pos = 0; pos < p->size(); pos++)
        {
          (*p)[pos] = static_cast<unsigned long int>(S[pos]);
        }
      int res = SWIG_OK;
      if (SWIG_IsOK(res)) {
        if (val) *val = p;
      }
      UNPROTECT(1);
      return res;
    }
  };

    template <>
  struct traits_asptr < std::vector<long int> > {
    static int asptr(SEXP obj, std::vector<long int> **val) {
      std::vector<long int> *p;
      // not sure how to check the size of the SEXP obj is correct
      int sexpsz = Rf_length(obj);
      p = new std::vector<long int>(sexpsz);
      SEXP coerced;
      PROTECT(coerced = Rf_coerceVector(obj, INTSXP));
      int *S = INTEGER_POINTER(coerced);
      for (unsigned pos = 0; pos < p->size(); pos++)
        {
          (*p)[pos] = static_cast<long int>(S[pos]);
        }
      int res = SWIG_OK;
      if (SWIG_IsOK(res)) {
        if (val) *val = p;
      }
      UNPROTECT(1);
      return res;
    }
  };

#else
    // 64 bit integer types
    template <>
  struct traits_asptr < std::vector<unsigned long long int> > {
    static int asptr(SEXP obj, std::vector<unsigned long long int> **val) {
      std::vector<unsigned long long int> *p;
      unsigned int sexpsz = Rf_length(obj);
      p = new std::vector<unsigned long long int>(sexpsz);
      SEXP coerced;
      PROTECT(coerced = Rf_coerceVector(obj, INTSXP));
      int *S = INTEGER_POINTER(coerced);
      for (unsigned pos = 0; pos < p->size(); pos++)
        {
          (*p)[pos] = static_cast<unsigned long long int>(S[pos]);
        }
      int res = SWIG_OK;
      if (SWIG_IsOK(res)) {
        if (val) *val = p;
      }
      UNPROTECT(1);
      return res;
    }
  };

    template <>
  struct traits_asptr < std::vector<long long int> > {
    static int asptr(SEXP obj, std::vector<long long int> **val) {
      std::vector<long long int> *p;
      // not sure how to check the size of the SEXP obj is correct
      int sexpsz = Rf_length(obj);
      p = new std::vector<long long int>(sexpsz);
      SEXP coerced;
      PROTECT(coerced = Rf_coerceVector(obj, INTSXP));
      int *S = INTEGER_POINTER(coerced);
      for (unsigned pos = 0; pos < p->size(); pos++)
        {
          (*p)[pos] = static_cast<long long int>(S[pos]);
        }
      int res = SWIG_OK;
      if (SWIG_IsOK(res)) {
        if (val) *val = p;
      }
      UNPROTECT(1);
      return res;
    }
  };

#endif

    template <>
  struct traits_asptr < std::vector<bool> > {
    static int asptr(SEXP obj, std::vector<bool> **val) {
      std::vector<bool> *p;
      // not sure how to check the size of the SEXP obj is correct
      int sexpsz = Rf_length(obj);
      p = new std::vector<bool>(sexpsz);
      SEXP coerced;
      PROTECT(coerced = Rf_coerceVector(obj, LGLSXP));
      int *S = LOGICAL_POINTER(coerced);
      for (unsigned pos = 0; pos < p->size(); pos++)
        {
          (*p)[pos] = static_cast<bool>(S[pos]);
        }
      int res = SWIG_OK;
      if (SWIG_IsOK(res)) {
        if (val) *val = p;
      }
      UNPROTECT(1);
      return res;
    }
  };

    template <>
      struct traits_asptr < std::vector<std::basic_string<char> > > {
      static int asptr(SEXP obj, std::vector<std::basic_string<char> > **val) {
	std::vector<std::basic_string<char> > *p;
      // R character vectors are STRSXP containing CHARSXP
      // access a CHARSXP using STRING_ELT
      int sexpsz = Rf_length(obj);
      p = new std::vector<std::basic_string<char> >(sexpsz);
      SEXP coerced;
      PROTECT(coerced = Rf_coerceVector(obj, STRSXP));
      //SEXP *S = CHARACTER_POINTER(coerced);
      for (unsigned pos = 0; pos < p->size(); pos++)
        {
	  const char * thecstring = CHAR(STRING_ELT(coerced, pos));
          (*p)[pos] = std::basic_string<char>(thecstring);
        }
      int res = SWIG_OK;
      if (SWIG_IsOK(res)) {
        if (val) *val = p;
      }
      UNPROTECT(1);
      return res;
    }
  };

    // catchall for R to vector conversion
  template <typename T>
  struct traits_asptr < std::vector<T> > {
    static int asptr(SEXP obj, std::vector<T> **val) {
      std::vector<T> *p;
      int res = SWIG_R_ConvertPtr(obj, (void**)&p, type_info< std::vector<T> >(), 0);
      if (SWIG_IsOK(res)) {
        if (val) *val = p;
      }
      return res;
    }
  };

  // now for vectors of vectors. These will be represented as lists of vectors on the
  // catch all that does everything with vectors
  template <>
    struct traits_from_ptr<std::vector<std::vector<unsigned int> > > {
      static SEXP from (std::vector< std::vector<unsigned int> > *val, int owner = 0) {
        SEXP result;
        // allocate the R list
        PROTECT(result = Rf_allocVector(VECSXP, val->size()));
        for (unsigned pos = 0; pos < val->size(); pos++)
          {
            // allocate the R vector
            SET_VECTOR_ELT(result, pos, Rf_allocVector(INTSXP, val->at(pos).size()));
            // Fill the R vector
            for (unsigned vpos = 0; vpos < val->at(pos).size(); ++vpos)
              {
                INTEGER_POINTER(VECTOR_ELT(result, pos))[vpos] = static_cast<int>(val->at(pos).at(vpos));
              }
          }
        UNPROTECT(1);
        return(result);
      }
    };


  template <>
    struct traits_from_ptr<std::vector<std::vector<int> > > {
      static SEXP from (std::vector< std::vector<int > > *val, int owner = 0) {
        SEXP result;
        // allocate the R list
        PROTECT(result = Rf_allocVector(VECSXP, val->size()));
        for (unsigned pos = 0; pos < val->size(); pos++)
          {
            // allocate the R vector
            SET_VECTOR_ELT(result, pos, Rf_allocVector(INTSXP, val->at(pos).size()));
            // Fill the R vector
            for (unsigned vpos = 0; vpos < val->at(pos).size(); ++vpos)
              {
                INTEGER_POINTER(VECTOR_ELT(result, pos))[vpos] = static_cast<int>(val->at(pos).at(vpos));
              }
          }
        UNPROTECT(1);
        return(result);
      }
    };

  template <>
    struct traits_from_ptr<std::vector<std::vector<float> > > {
      static SEXP from (std::vector< std::vector<float > > *val, int owner = 0) {
        SEXP result;
        // allocate the R list
        PROTECT(result = Rf_allocVector(VECSXP, val->size()));
        for (unsigned pos = 0; pos < val->size(); pos++)
          {
            // allocate the R vector
            SET_VECTOR_ELT(result, pos, Rf_allocVector(REALSXP, val->at(pos).size()));
            // Fill the R vector
            for (unsigned vpos = 0; vpos < val->at(pos).size(); ++vpos)
              {
                NUMERIC_POINTER(VECTOR_ELT(result, pos))[vpos] = static_cast<double>(val->at(pos).at(vpos));
              }
          }
        UNPROTECT(1);
        return(result);
      }
    };

  template <>
    struct traits_from_ptr<std::vector<std::vector<double> > > {
      static SEXP from (std::vector< std::vector<double > > *val, int owner = 0) {
        SEXP result;
        // allocate the R list
        PROTECT(result = Rf_allocVector(VECSXP, val->size()));
        for (unsigned pos = 0; pos < val->size(); pos++)
          {
            // allocate the R vector
            SET_VECTOR_ELT(result, pos, Rf_allocVector(REALSXP, val->at(pos).size()));
            // Fill the R vector
            for (unsigned vpos = 0; vpos < val->at(pos).size(); ++vpos)
              {
                NUMERIC_POINTER(VECTOR_ELT(result, pos))[vpos] = static_cast<double>(val->at(pos).at(vpos));
              }
          }
        UNPROTECT(1);
        return(result);
      }
    };

  template <>
    struct traits_from_ptr<std::vector<std::vector<bool> > > {
      static SEXP from (std::vector< std::vector<bool> > *val, int owner = 0) {
        SEXP result;
        // allocate the R list
        PROTECT(result = Rf_allocVector(VECSXP, val->size()));
        for (unsigned pos = 0; pos < val->size(); pos++)
          {
            // allocate the R vector
            SET_VECTOR_ELT(result, pos, Rf_allocVector(LGLSXP, val->at(pos).size()));
            // Fill the R vector
            for (unsigned vpos = 0; vpos < val->at(pos).size(); ++vpos)
              {
                LOGICAL_POINTER(VECTOR_ELT(result, pos))[vpos] = (val->at(pos).at(vpos));
              }
          }
        UNPROTECT(1);
        return(result);
      }
    };

  template <typename T>
    struct traits_from_ptr< std::vector < std::vector< T > > > {
    static SEXP from (std::vector < std::vector< T > > *val, int owner = 0) {
      return SWIG_R_NewPointerObj(val, type_info< std::vector < std::vector< T > > >(), owner);
    }
  };

  /////////////////////////////////////////////////////////////////

  // R side
  template <>
    struct traits_asptr < std::vector< std::vector<unsigned int> > > {
    static int asptr(SEXP obj, std::vector< std::vector<unsigned int> > **val) {
      std::vector <std::vector<unsigned int> > *p;
      // this is the length of the list
      unsigned int sexpsz = Rf_length(obj);
      p = new std::vector< std::vector<unsigned int> > (sexpsz);

      for (unsigned listpos = 0; listpos < sexpsz; ++listpos)
        {
          unsigned vecsize = Rf_length(VECTOR_ELT(obj, listpos));
          for (unsigned vpos = 0; vpos < vecsize; ++vpos)
            {
              (*p)[listpos].push_back(static_cast<int>(INTEGER_POINTER(VECTOR_ELT(obj, listpos))[vpos]));
            }
        }

      int res = SWIG_OK;

      if (SWIG_IsOK(res)) {
        if (val) *val = p;
      }
      return res;
    }
  };

  template <>
    struct traits_asptr < std::vector< std::vector< int> > > {
    static int asptr(SEXP obj, std::vector< std::vector< int> > **val) {
      std::vector <std::vector< int> > *p;
      // this is the length of the list
      unsigned int sexpsz = Rf_length(obj);
      p = new std::vector< std::vector< int> > (sexpsz);

      for (unsigned listpos = 0; listpos < sexpsz; ++listpos)
        {
          unsigned vecsize = Rf_length(VECTOR_ELT(obj, listpos));
          for (unsigned vpos = 0; vpos < vecsize; ++vpos)
            {
              (*p)[listpos].push_back(static_cast<int>(INTEGER_POINTER(VECTOR_ELT(obj, listpos))[vpos]));
            }
        }

      int res = SWIG_OK;

      if (SWIG_IsOK(res)) {
        if (val) *val = p;
      }
      return res;
    }
  };

  template <>
    struct traits_asptr < std::vector< std::vector< float> > > {
    static int asptr(SEXP obj, std::vector< std::vector< float> > **val) {
      std::vector <std::vector< float> > *p;
      // this is the length of the list
      unsigned int sexpsz = Rf_length(obj);
      p = new std::vector< std::vector< float> > (sexpsz);

      for (unsigned listpos = 0; listpos < sexpsz; ++listpos)
        {
          unsigned vecsize = Rf_length(VECTOR_ELT(obj, listpos));
          for (unsigned vpos = 0; vpos < vecsize; ++vpos)
            {
              (*p)[listpos].push_back(static_cast<float>(NUMERIC_POINTER(VECTOR_ELT(obj, listpos))[vpos]));
            }
        }

      int res = SWIG_OK;

      if (SWIG_IsOK(res)) {
        if (val) *val = p;
      }
      return res;
    }
  };

  template <>
    struct traits_asptr < std::vector< std::vector< double> > > {
    static int asptr(SEXP obj, std::vector< std::vector< double> > **val) {
      std::vector <std::vector< double> > *p;
      // this is the length of the list
      unsigned int sexpsz = Rf_length(obj);
      p = new std::vector< std::vector< double> > (sexpsz);

      for (unsigned listpos = 0; listpos < sexpsz; ++listpos)
        {
          unsigned vecsize = Rf_length(VECTOR_ELT(obj, listpos));
          for (unsigned vpos = 0; vpos < vecsize; ++vpos)
            {
              (*p)[listpos].push_back(static_cast<double>(NUMERIC_POINTER(VECTOR_ELT(obj, listpos))[vpos]));
            }
        }

      int res = SWIG_OK;

      if (SWIG_IsOK(res)) {
        if (val) *val = p;
      }
      return res;
    }
  };

  template <>
    struct traits_asptr < std::vector< std::vector< bool > > > {
    static int asptr(SEXP obj, std::vector< std::vector< bool> > **val) {
      std::vector <std::vector< bool > > *p;
      // this is the length of the list
      unsigned int sexpsz = Rf_length(obj);
      p = new std::vector< std::vector< bool > > (sexpsz);

      for (unsigned listpos = 0; listpos < sexpsz; ++listpos)
        {
          unsigned vecsize = Rf_length(VECTOR_ELT(obj, listpos));
          for (unsigned vpos = 0; vpos < vecsize; ++vpos)
            {
              (*p)[listpos].push_back(static_cast<bool>(LOGICAL_POINTER(VECTOR_ELT(obj, listpos))[vpos]));
            }
        }

      int res = SWIG_OK;

      if (SWIG_IsOK(res)) {
        if (val) *val = p;
      }
      return res;
    }
  };

  //  catchall
  template <typename T>
    struct traits_asptr < std::vector< std::vector<T> > > {
    static int asptr(SEXP obj, std::vector< std::vector<T> > **val) {
      std::vector< std::vector<T> > *p;
      Rprintf("vector of vectors - unsupported content\n");
      int res = SWIG_R_ConvertPtr(obj, (void**)&p, type_info< std::vector< std::vector<T> > > (), 0);
      if (SWIG_IsOK(res)) {
        if (val) *val = p;
      }
      return res;
    }
  };

  }
%}

#define %swig_vector_methods(Type...) %swig_sequence_methods(Type)
#define %swig_vector_methods_val(Type...) %swig_sequence_methods_val(Type);

%define %traits_type_name(Type...)
%fragment(SWIG_Traits_frag(Type), "header",
          fragment="StdTraits",fragment="StdVectorTraits") {
  namespace swig {
    template <>  struct traits< Type > {
      typedef pointer_category category;
      static const char* type_name() {
        return #Type;
      }
    };
  }
 }
%enddef

%include <std/std_vector.i>

%typemap_traits_ptr(SWIG_TYPECHECK_VECTOR, std::vector<double>)
%traits_type_name(std::vector<double>)
%typemap("rtypecheck") std::vector<double>, std::vector<double> *, std::vector<double> &
    %{ is.numeric($arg) %}
%typemap("rtype") std::vector<double> "numeric"
%typemap("scoercein") std::vector<double>, std::vector<double> *, std::vector<double> & "$input = as.numeric($input);";

%typemap_traits_ptr(SWIG_TYPECHECK_VECTOR, std::vector<float>)
%traits_type_name(std::vector<float>)

// reuse these for float
%typemap("rtype") std::vector<float>, std::vector<float> *, std::vector<float> & = std::vector<double>;
%typemap("rtypecheck") std::vector<float>, std::vector<float> *, std::vector<float> & = std::vector<double>;
%typemap("scoercein") std::vector<float>, std::vector<float> *, std::vector<float> & = std::vector<double>;


%typemap_traits_ptr(SWIG_TYPECHECK_VECTOR, std::vector<bool>);
%traits_type_name(std::vector<bool>);
%typemap("rtypecheck") std::vector<bool>, std::vector<bool> *, std::vector<bool> &
   %{ is.logical($arg) %}
%typemap("rtype") std::vector<bool> "logical"
%typemap("scoercein") std::vector<bool> , std::vector<bool> & "$input = as.logical($input);";


%typemap_traits_ptr(SWIG_TYPECHECK_VECTOR, std::vector<int>);
%traits_type_name(std::vector<int>);
%typemap("rtypecheck") std::vector<int>, std::vector<int>  *, std::vector<int>  &
   %{ is.integer($arg) || is.numeric($arg) %}

%typemap("rtype") std::vector<int> "integer"
%typemap("scoercein") std::vector<int> , std::vector<int> *, std::vector<int> & "$input = as.integer($input);";

// strings
%typemap("rtype") std::vector< std::basic_string<char> >,  
std::vector< std::basic_string<char> > *,
   std::vector< std::basic_string<char> > & "character"

%typemap("rtypecheck") std::vector< std::basic_string<char> >,  
std::vector< std::basic_string<char> > *,
   std::vector< std::basic_string<char> > &
   %{ is.character($arg) %}

%typemap("scoercein") std::vector< std::basic_string<char> >,  
std::vector< std::basic_string<char> > *,
   std::vector< std::basic_string<char> > & "$input = as.character($input);";

%typemap("scoerceout") std::vector< std::basic_string<char> >,  
std::vector< std::basic_string<char> > *,
   std::vector< std::basic_string<char> > & 
%{    %}

%apply std::vector< std::basic_string<char> > { std::vector< std::string> };

// all the related integer vectors
// signed
%typemap_traits_ptr(SWIG_TYPECHECK_VECTOR, std::vector<signed char>);
%traits_type_name(std::vector<signed char>);

%typemap_traits_ptr(SWIG_TYPECHECK_VECTOR, std::vector<signed short>);
%traits_type_name(std::vector<signed short>);

#if defined(SWIGWORDSIZE64)
%typemap_traits_ptr(SWIG_TYPECHECK_VECTOR, std::vector<long int>);
%traits_type_name(std::vector<long int>);
#else
%typemap_traits_ptr(SWIG_TYPECHECK_VECTOR, std::vector<long long int>);
%traits_type_name(std::vector<long long int>);
#endif

// unsigned
%typemap_traits_ptr(SWIG_TYPECHECK_VECTOR, std::vector<unsigned char>);
%traits_type_name(std::vector<unsigned char>);

%typemap_traits_ptr(SWIG_TYPECHECK_VECTOR, std::vector<unsigned short>);
%traits_type_name(std::vector<unsigned short>);

%typemap_traits_ptr(SWIG_TYPECHECK_VECTOR, std::vector<unsigned int>);
%traits_type_name(std::vector<unsigned int>);

#if defined(SWIGWORDSIZE64)
%typemap_traits_ptr(SWIG_TYPECHECK_VECTOR, std::vector<unsigned long int>);
%traits_type_name(std::vector<unsigned long int>);
#else
%typemap_traits_ptr(SWIG_TYPECHECK_VECTOR, std::vector<unsigned long long int>);
%traits_type_name(std::vector<unsigned long long int>);
#endif

// These R side typemaps are common for integer types
// but we can't use %apply as it will copy the C side ones too
// Also note that we don't seem to be able to use types like
// int_least8_t here.
%typemap("rtype") std::vector<signed char>, std::vector<signed char> *, std::vector<signed char> & = std::vector<int>;
%typemap("rtype") std::vector<signed short>, std::vector<signed short> *, std::vector<signed short> & = std::vector<int>;
%typemap("rtype") std::vector<unsigned char>, std::vector<unsigned char> *, std::vector<unsigned char> & = std::vector<int>;
%typemap("rtype") std::vector<unsigned int>, std::vector<unsigned int> *, std::vector<unsigned int> & = std::vector<int>;

#if defined(SWIGWORDSIZE64)
%typemap("rtype") std::vector<long int>, std::vector<long int> *, std::vector<long int> & = std::vector<int>;
%typemap("rtype") std::vector<unsigned long int>, std::vector<unsigned long int> *, std::vector<unsigned long int> & = std::vector<int>;
#else
%typemap("rtype") std::vector<long long int>, std::vector<long long int> *, std::vector<long long int> & = std::vector<int>;
%typemap("rtype") std::vector<unsigned long long int>, std::vector<unsigned long long int> *, std::vector<unsigned long long int> & = std::vector<int>;
#endif


%typemap("scoercein") std::vector<signed char>, std::vector<signed char> *, std::vector<signed char> & = std::vector<int>;
%typemap("scoercein") std::vector<signed short>, std::vector<signed short> *, std::vector<signed short> & = std::vector<int>;
%typemap("scoercein") std::vector<unsigned char>, std::vector<unsigned char> *, std::vector<unsigned char> & = std::vector<int>;
%typemap("scoercein") std::vector<unsigned int>, std::vector<unsigned int> *, std::vector<unsigned int> & = std::vector<int>;

#if defined(SWIGWORDSIZE64)
%typemap("scoercein") std::vector<long int>, std::vector<long int> *, std::vector<long int> & = std::vector<int>;
%typemap("scoercein") std::vector<unsigned long int>, std::vector<unsigned long int> *, std::vector<unsigned long int> & = std::vector<int>;
#else
%typemap("scoercein") std::vector<long long int>, std::vector<long long int> *, std::vector<long long int> & = std::vector<int>;
%typemap("scoercein") std::vector<unsigned long long int>, std::vector<unsigned long long int> *, std::vector<unsigned long long int> & = std::vector<int>;
#endif

%typemap("rtypecheck") std::vector<signed char>, std::vector<signed char> *, std::vector<signed char> & = std::vector<int>;
%typemap("rtypecheck") std::vector<signed short>, std::vector<signed short> *, std::vector<signed short> & = std::vector<int>;
%typemap("rtypecheck") std::vector<unsigned char>, std::vector<unsigned char> *, std::vector<unsigned char> & = std::vector<int>;
%typemap("rtypecheck") std::vector<unsigned int>, std::vector<unsigned int> *, std::vector<unsigned int> & = std::vector<int>;

#if defined(SWIGWORDSIZE64)
%typemap("rtypecheck") std::vector<long int>, std::vector<long int> *, std::vector<long int> &  = std::vector<int>;
%typemap("rtypecheck") std::vector<unsigned long int>, std::vector<unsigned long int> *, std::vector<unsigned long int> & = std::vector<int>;
#else
%typemap("rtypecheck") std::vector<long long int>, std::vector<long long int> *, std::vector<long long int> & = std::vector<int>;
%typemap("rtypecheck") std::vector<unsigned long long int>, std::vector<unsigned long long int> *, std::vector<unsigned long long int> & = std::vector<int>;
#endif

///////////////////////////////////////////////////////////////

%typemap_traits_ptr(SWIG_TYPECHECK_VECTOR, std::vector<std::vector<int> >);
%traits_type_name(std::vector< std::vector<int> >);
%typemap("rtypecheck") std::vector<std::vector<int> >, std::vector<std::vector<int> > *, std::vector<std::vector<int> > &
   %{ is.list($arg) && all(sapply($arg , is.integer) || sapply($arg, is.numeric)) %}
%typemap("rtype") std::vector<std::vector<int> >, std::vector<std::vector<int> > *, std::vector<std::vector<int> > & "list"
%typemap("scoercein") std::vector< std::vector<int> >, std::vector<std::vector<int> > *, std::vector<std::vector<int> > & "$input = lapply($input, as.integer);";

%typemap_traits_ptr(SWIG_TYPECHECK_VECTOR, std::vector<std::vector<unsigned int> >);
%traits_type_name(std::vector< std::vector<unsigned int> >);
%typemap("rtypecheck") std::vector<std::vector<unsigned int> >, std::vector<std::vector<unsigned int> > *, std::vector<std::vector<unsigned int> > &
   %{ is.list($arg) && all(sapply($arg , is.integer) || sapply($arg, is.numeric)) %}
%typemap("rtype") std::vector<std::vector<unsigned int> >, std::vector<std::vector<unsigned int> > *, std::vector<std::vector<unsigned int> > & "list"
%typemap("scoercein") std::vector< std::vector<unsigned int> >, std::vector<std::vector<int> > *, std::vector<std::vector<unsigned int> > & "$input = lapply($input, as.integer);";

%typemap_traits_ptr(SWIG_TYPECHECK_VECTOR, std::vector<std::vector<float> >);
%traits_type_name(std::vector< std::vector<float> >);
%typemap("rtypecheck") std::vector<std::vector<float> >, std::vector<std::vector<float> > *, std::vector<std::vector<float> > &
   %{ is.list($arg) && all(sapply($arg , is.integer) || sapply($arg, is.numeric)) %}
%typemap("rtype") std::vector<std::vector<float> >, std::vector<std::vector<float> > *, std::vector<std::vector<float> > "list"
%typemap("scoercein") std::vector< std::vector<float> >, std::vector<std::vector<float> > *, std::vector<std::vector<float> > & "$input = lapply($input, as.numeric);";

%typemap_traits_ptr(SWIG_TYPECHECK_VECTOR, std::vector<std::vector<double> >);
%traits_type_name(std::vector< std::vector<double> >);
%typemap("rtypecheck") std::vector<std::vector<double> >, std::vector<std::vector<double> > *, std::vector<std::vector<double> > &
   %{ is.list($arg) && all(sapply($arg , is.integer) || sapply($arg, is.numeric)) %}
%typemap("rtype") std::vector<std::vector<double> >, std::vector<std::vector<double> > *, std::vector<std::vector<double> > & "list"
%typemap("scoercein") std::vector< std::vector<double> >, std::vector<std::vector<double> > *, std::vector<std::vector<double> > &
 "$input = lapply($input, as.numeric);";

%typemap_traits_ptr(SWIG_TYPECHECK_VECTOR, std::vector<std::vector<bool> >);
%traits_type_name(std::vector< std::vector<bool> >);
%typemap("rtypecheck") std::vector<std::vector<bool> >, std::vector<std::vector<bool> > *, std::vector<std::vector<bool> > &
   %{ is.list($arg) && all(sapply($arg , is.integer) || sapply($arg, is.numeric)) %}
%typemap("rtype") std::vector<std::vector<bool> >, std::vector<std::vector<bool> > *, std::vector<std::vector<bool> > & "list"
%typemap("scoercein") std::vector< std::vector<bool> >, std::vector<std::vector<bool> > *, std::vector<std::vector<bool> > & "$input = lapply($input, as.logical);";

// we don't want these to be given R classes as they
// have already been turned into R vectors.
%typemap(scoerceout) std::vector<double>,
   std::vector<double>*,
   std::vector<double>&,
   std::vector<float> ,
   std::vector<float>*,
   std::vector<float> ,
   std::vector<signed char>,
   std::vector<signed char>*,
   std::vector<signed char>&,
   std::vector<signed short>,
   std::vector<signed short>*,
   std::vector<signed short>&,
   std::vector<int>,
   std::vector<int>*,
   std::vector<int>&,
   std::vector<unsigned char>,
   std::vector<unsigned char>*,
   std::vector<unsigned char>&,
   std::vector<unsigned short>,
   std::vector<unsigned short>*,
   std::vector<unsigned short>&,
   std::vector<unsigned int>,
   std::vector<unsigned int>*,
   std::vector<unsigned int>&,
   std::vector<bool>,
   std::vector<bool>*,
   std::vector<bool>&,
 // vectors of vectors
   std::vector< std::vector<unsigned int> >,
   std::vector< std::vector<unsigned int> >*,
   std::vector< std::vector<unsigned int> >&,
   std::vector< std::vector<int> >,
   std::vector< std::vector<int> >*,
   std::vector< std::vector<int> >&,
   std::vector< std::vector<float> >,
   std::vector< std::vector<float> >*,
   std::vector< std::vector<float> >&,
   std::vector< std::vector<double> >,
   std::vector< std::vector<double> >*,
   std::vector< std::vector<double> >&,
   std::vector< std::vector<bool> >,
   std::vector< std::vector<bool> >*,
   std::vector< std::vector<bool> >&
 %{    %}

#if defined(SWIGWORDSIZE64)
%typemap(scoerceout) std::vector<long int>,
   std::vector<long int>*,
   std::vector<long int>&,
   std::vector<unsigned long int>,
   std::vector<unsigned long int>*,
   std::vector<unsigned long int>&
 %{    %}
#else

%typemap(scoerceout) std::vector<long long int>,
   std::vector<long long int>*,
   std::vector<long long int>&,
   std::vector<unsigned long long int>,
   std::vector<unsigned long long int>*,
   std::vector<unsigned long long int>&
 %{    %}

#endif
