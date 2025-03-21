
# for compatibility between Cython 0.29.x and Cython 3.0.x
# TODO: remove after final switch to Cython 3.x+
target_compile_options(_catboost PRIVATE
  "-DCYTHON_EXTERN_C=extern \"C\""
)
