
# Unfortunately, standard FindPython3 CMake module does not support Interpreter and NumPy COMPONENTS
# from different Python installations that prevents CI building for multiple Python versions and
# Cross-compilation, so we'll handle these special cases by setting ${Python3_NumPy_INCLUDE_DIRS} explicitly
if(Python3_NumPy_INCLUDE_DIR)
  if(EXISTS "${Python3_NumPy_INCLUDE_DIR}" AND IS_DIRECTORY "${Python3_NumPy_INCLUDE_DIR}")
    set(Python3_NumPy_INCLUDE_DIRS ${Python3_NumPy_INCLUDE_DIR})
  else()
    message(FATAL_ERROR "Python3_NumPy_INCLUDE_DIR=\"${Python3_NumPy_INCLUDE_DIR}\" does not exist.")
  endif()
else()
  find_package(Python3 REQUIRED COMPONENTS
    Development NumPy
  )
endif()
