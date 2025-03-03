#define %alias %feature("alias")
#define %freefunc %feature("freefunc")
#define %markfunc %feature("markfunc")
#define %mixin %feature("mixin")
#define %predicate %feature("predicate", "1")
#define %bang %feature("bang", "1")
#define %trackobjects %feature("trackobjects")
#define %nooutput %feature("outputs","0")
#define %initstack %feature("initstack", "1")
#define %ignorestack %feature("initstack", "0")

/* ------------------------------------------------------------------------- */
/*
  Enable keywords parameters
*/

#define %kwargs      %feature("kwargs") 
#define %nokwargs    %feature("kwargs", "0")
#define %clearkwargs %feature("kwargs", "")

