#ifndef CSHARP_CSHARPKW_SWG_
#define CSHARP_CSHARPKW_SWG_

/* Warnings for C# keywords */
#define CSHARPKW(x) %keywordwarn("'" `x` "' is a C# keyword, renaming to '" `x` "_'",rename="%s_")  `x`

#define CSHARPCLASSKW(x) %keywordwarn("'" `x` "' is a special method name used in the C# wrapper classes, class renamed to '" `x` "_'",%$isclass,rename="%s_") `x`

/*
   from
   http://www.jaggersoft.com/csharp_grammar.html#1.7%20Keywords

*/

CSHARPKW(abstract);
CSHARPKW(as);
CSHARPKW(base);
CSHARPKW(bool);
CSHARPKW(break);
CSHARPKW(byte);
CSHARPKW(case);
CSHARPKW(catch);
CSHARPKW(char);
CSHARPKW(checked);
CSHARPKW(class);
CSHARPKW(const);
CSHARPKW(continue);
CSHARPKW(decimal);
CSHARPKW(default);
CSHARPKW(delegate);
CSHARPKW(do);
CSHARPKW(double);
CSHARPKW(else);
CSHARPKW(enum);
CSHARPKW(event);
CSHARPKW(explicit);
CSHARPKW(extern);
CSHARPKW(false);
CSHARPKW(finally);
CSHARPKW(fixed);
CSHARPKW(float);
CSHARPKW(for);
CSHARPKW(foreach);
CSHARPKW(goto);
CSHARPKW(if);
CSHARPKW(implicit);
CSHARPKW(in);
CSHARPKW(int);
CSHARPKW(interface);
CSHARPKW(internal);
CSHARPKW(is);
CSHARPKW(lock);
CSHARPKW(long);
CSHARPKW(namespace);
CSHARPKW(new);
CSHARPKW(null);
CSHARPKW(object);
CSHARPKW(operator);
CSHARPKW(out);
CSHARPKW(override);
CSHARPKW(params);
CSHARPKW(private);
CSHARPKW(protected);
CSHARPKW(public);
CSHARPKW(readonly);
CSHARPKW(ref);
CSHARPKW(return);
CSHARPKW(sbyte);
CSHARPKW(sealed);
CSHARPKW(short);
CSHARPKW(sizeof);
CSHARPKW(stackalloc);
CSHARPKW(static);
CSHARPKW(struct);
CSHARPKW(string);
CSHARPKW(switch);
CSHARPKW(this);
CSHARPKW(throw);
CSHARPKW(true);
CSHARPKW(try);
CSHARPKW(typeof);
CSHARPKW(uint);
CSHARPKW(ulong);
CSHARPKW(unchecked);
CSHARPKW(unsafe);
CSHARPKW(ushort);
CSHARPKW(using);
CSHARPKW(virtual);
CSHARPKW(void);
CSHARPKW(volatile);
CSHARPKW(while);

CSHARPCLASSKW(delete);

#undef CSHARPKW

#endif //CSHARP_CSHARPKW_SWG_
