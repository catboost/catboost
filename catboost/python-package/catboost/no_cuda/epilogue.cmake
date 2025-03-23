
# for compatibility between Cython 0.29.x and Cython 3.0.x
# TODO: remove after final switch to Cython 3.x+
target_compile_options(_catboost_no_cuda PRIVATE
  "-DCYTHON_EXTERN_C=extern \"C\""
)

target_include_directories(_catboost_no_cuda PRIVATE
  ${Python3_NumPy_INCLUDE_DIRS}
)

target_cython_include_directories(_catboost_no_cuda
  ${Python3_NumPy_INCLUDE_DIRS}
)
