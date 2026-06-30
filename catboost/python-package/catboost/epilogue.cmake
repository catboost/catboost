
target_include_directories(_catboost PRIVATE
  ${Python3_NumPy_INCLUDE_DIRS}
)

target_cython_include_directories(_catboost
  ${Python3_NumPy_INCLUDE_DIRS}
)
