require 'mkmf'

dir_config('yourlib')

if have_header('yourlib.h') and have_library('yourlib', 'yourlib_init')
  # If you use swig -c option, you may have to link libswigrb.
  # have_library('swigrb')
  create_makefile('yourlib')
end
