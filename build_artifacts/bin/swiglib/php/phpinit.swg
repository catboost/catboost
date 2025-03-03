
/* ------------------------------------------------------------
 * The start of the PHP initialization function
 * ------------------------------------------------------------ */

%insert(init) "swiginit.swg"

%init %{
SWIG_php_minit {
  SWIG_InitializeModule((void*)&module_number);
%}

%fragment("swig_php_init_member_ptr2", "header") %{
#define SWIG_MEMBER_PTR "CLASS::*"

static void swig_member_ptr_dtor(zend_resource *res) {
  efree(res->ptr);
}

static int swig_member_ptr = 0;
%}

%fragment("swig_php_init_member_ptr", "init", fragment="swig_php_init_member_ptr2") %{
  swig_member_ptr = zend_register_list_destructors_ex(swig_member_ptr_dtor, NULL, SWIG_MEMBER_PTR, module_number);
%}
