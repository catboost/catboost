#ifndef JAVA_JAVAKW_SWG_
#define JAVA_JAVAKW_SWG_

/* Warnings for Java keywords */
#define JAVAKW(x) %keywordwarn("'" `x` "' is a java keyword",rename="_%s")  `x`

/*
   from
   http://java.sun.com/docs/books/tutorial/java/nutsandbolts/_keywords.html 
*/

JAVAKW(abstract);
JAVAKW(double);
JAVAKW(int);
JAVAKW(strictfp);
JAVAKW(boolean);
JAVAKW(else);
JAVAKW(interface);
JAVAKW(super);
JAVAKW(break);
JAVAKW(extends);	
JAVAKW(long);	
JAVAKW(switch);
JAVAKW(byte);	
JAVAKW(final);	
JAVAKW(native);	
JAVAKW(synchronized);
JAVAKW(case);	
JAVAKW(finally);	
JAVAKW(new);	
JAVAKW(this);
JAVAKW(catch);	
JAVAKW(float);	
JAVAKW(package);	
JAVAKW(throw);
JAVAKW(char);	
JAVAKW(for);	
JAVAKW(private);	
JAVAKW(throws);
JAVAKW(class);	
JAVAKW(goto); 	
JAVAKW(protected);	
JAVAKW(transient);
JAVAKW(const); 	
JAVAKW(if);	
JAVAKW(public);	
JAVAKW(try);
JAVAKW(continue);	
JAVAKW(implements);	
JAVAKW(return);	
JAVAKW(void);
JAVAKW(default);	
JAVAKW(import);	
JAVAKW(short);	
JAVAKW(volatile);
JAVAKW(do);	
JAVAKW(instanceof);	
JAVAKW(static);	
JAVAKW(while);


/* others bad names */

/* Note here that only *::clone() is bad, and *::clone(int) is ok */
%namewarn("321:clone() is a java bad method name") *::clone();


#undef JAVAKW

#endif //JAVA_JAVAKW_SWG_
