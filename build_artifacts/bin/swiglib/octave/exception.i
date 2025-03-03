%include <typemaps/exception.swg>

%insert("runtime") {
  %define_as(SWIG_exception(code, msg), %block(%error(code, msg); SWIG_fail; ))
}

%define SWIG_RETHROW_OCTAVE_EXCEPTIONS
  /* rethrow any exceptions thrown by Octave */
%#if SWIG_OCTAVE_PREREQ(4,2,0)
  catch (octave::execution_exception& _e) { throw; }
  catch (octave::exit_exception& _e) { throw; }
  catch (octave::interrupt_exception& _e) { throw; }
%#endif
%enddef
