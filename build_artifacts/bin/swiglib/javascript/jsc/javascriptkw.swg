#ifndef JAVASCRIPT_JAVASCRIPTKW_SWG_
#define JAVASCRIPT_JAVASCRIPTKW_SWG_

/* Warnings for Java keywords */
#define JAVASCRIPTKW(x) %keywordwarn("'" `x` "' is a javascript keyword, renaming to '_"`x`"'",rename="_%s")  `x`

/* Taken from https://developer.mozilla.org/En/Core_JavaScript_1.5_Reference/Reserved_Words */

JAVASCRIPTKW(break);
JAVASCRIPTKW(case);
JAVASCRIPTKW(catch);
JAVASCRIPTKW(continue);
JAVASCRIPTKW(default);
JAVASCRIPTKW(delete);
JAVASCRIPTKW(do);
JAVASCRIPTKW(else);
JAVASCRIPTKW(finally);
JAVASCRIPTKW(for);
JAVASCRIPTKW(function);
JAVASCRIPTKW(if);
JAVASCRIPTKW(in);
JAVASCRIPTKW(instanceof);
JAVASCRIPTKW(new);
JAVASCRIPTKW(return);
JAVASCRIPTKW(switch);
JAVASCRIPTKW(this);
JAVASCRIPTKW(throw);
JAVASCRIPTKW(try);
JAVASCRIPTKW(typeof);
JAVASCRIPTKW(var);
JAVASCRIPTKW(void);
JAVASCRIPTKW(while);
JAVASCRIPTKW(with);

/* others bad names if any*/
// for example %namewarn("321:clone() is a javascript bad method name") *::clone();

#undef JAVASCRIPTKW

#endif //JAVASCRIPT_JAVASCRIPTKW_SWG_
