
/* Export the SWIG initialization function */
%header %{
#ifdef __cplusplus
extern "C"
#endif
#ifndef MULTIPLICITY
SWIGEXPORT void SWIG_init (CV* cv);
#else
SWIGEXPORT void SWIG_init (pTHXo_ CV* cv);
#endif
%}

/* Module initialization function */

%insert(init) "swiginit.swg"

%init %{

#if defined(__cplusplus) && ! defined(XSPROTO)
extern "C"
#endif

XS(SWIG_init) {
    dXSARGS;
    int i;
    (void)items;

    SWIG_InitializeModule(0);

    /* Install commands */
    for (i = 0; swig_commands[i].name; i++) {
      /* Casts only needed for Perl < 5.10. */
#ifdef __cplusplus
      newXS(const_cast<char*>(swig_commands[i].name), swig_commands[i].wrapper, const_cast<char*>(__FILE__));
#else
      newXS((char*)swig_commands[i].name, swig_commands[i].wrapper, (char*)__FILE__);
#endif
    }

    /* Install variables */
    for (i = 0; swig_variables[i].name; i++) {
      SV *sv;
      sv = get_sv(swig_variables[i].name, TRUE | 0x2 | GV_ADDMULTI);
      if (swig_variables[i].type) {
	SWIG_MakePtr(sv,(void *)1, *swig_variables[i].type,0);
      } else {
	sv_setiv(sv,(IV) 0);
      }
      swig_create_magic(sv, swig_variables[i].name, swig_variables[i].set, swig_variables[i].get); 
    }

    /* Install constant */
    for (i = 0; swig_constants[i].type; i++) {
      SV *sv;
      sv = get_sv(swig_constants[i].name, TRUE | 0x2 | GV_ADDMULTI);
      switch(swig_constants[i].type) {
      case SWIG_INT:
	sv_setiv(sv, (IV) swig_constants[i].lvalue);
	break;
      case SWIG_FLOAT:
	sv_setnv(sv, (double) swig_constants[i].dvalue);
	break;
      case SWIG_STRING:
	sv_setpv(sv, (const char *) swig_constants[i].pvalue);
	break;
      case SWIG_POINTER:
	SWIG_MakePtr(sv, swig_constants[i].pvalue, *(swig_constants[i].ptype),0);
	break;
      case SWIG_BINARY:
	SWIG_MakePackedObj(sv, swig_constants[i].pvalue, swig_constants[i].lvalue, *(swig_constants[i].ptype));
	break;
      default:
	break;
      }
      SvREADONLY_on(sv);
    }
%}
