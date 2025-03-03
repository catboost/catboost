 #define %scilabconst(flag) %feature("scilab:const","flag")

// Create Scilab variable
#define SWIG_CreateScilabVariable_frag(Type...) %fragment_name(CreateScilabVariable, Type)
#define SWIG_CreateScilabVariable_dec(Type...) %symbol_name(CreateScilabVariable, Type)
