#ifndef D_DKW_SWG_
#define D_DKW_SWG_

/* Warnings for D keywords */
#define DKEYWORD(x) %keywordwarn("'" `x` "' is a D keyword, renaming to '_" `x` "'",rename="_%s")  `x`

// Source: http://www.digitalmars.com/d/{1.0,2.0}/lex.html and
DKEYWORD(Error);
DKEYWORD(Exception);
DKEYWORD(Object);
DKEYWORD(__FILE__);
DKEYWORD(__LINE__);
DKEYWORD(__gshared);
DKEYWORD(__thread);
DKEYWORD(__traits);
DKEYWORD(abstract);
DKEYWORD(alias);
DKEYWORD(align);
DKEYWORD(asm);
DKEYWORD(assert);
DKEYWORD(auto);
DKEYWORD(body);
DKEYWORD(bool);
DKEYWORD(break);
DKEYWORD(byte);
DKEYWORD(case);
DKEYWORD(cast);
DKEYWORD(catch);
DKEYWORD(cdouble);
DKEYWORD(cent);
DKEYWORD(cfloat);
DKEYWORD(char);
DKEYWORD(class);
DKEYWORD(const);
DKEYWORD(continue);
DKEYWORD(creal);
DKEYWORD(dchar);
DKEYWORD(debug);
DKEYWORD(default);
DKEYWORD(delegate);
DKEYWORD(delete);
DKEYWORD(deprecated);
DKEYWORD(do);
DKEYWORD(double);
DKEYWORD(dstring);
DKEYWORD(else);
DKEYWORD(enum);
DKEYWORD(export);
DKEYWORD(extern);
DKEYWORD(false);
DKEYWORD(final);
DKEYWORD(finally);
DKEYWORD(float);
DKEYWORD(for);
DKEYWORD(foreach);
DKEYWORD(foreach_reverse);
DKEYWORD(function);
DKEYWORD(goto);
DKEYWORD(idouble);
DKEYWORD(if);
DKEYWORD(ifloat);
DKEYWORD(immutable);
DKEYWORD(import);
DKEYWORD(in);
DKEYWORD(inout);
DKEYWORD(int);
DKEYWORD(interface);
DKEYWORD(invariant);
DKEYWORD(ireal);
DKEYWORD(is);
DKEYWORD(lazy);
DKEYWORD(long);
DKEYWORD(macro);
DKEYWORD(mixin);
DKEYWORD(module);
DKEYWORD(new);
DKEYWORD(nothrow);
DKEYWORD(null);
DKEYWORD(out);
DKEYWORD(override);
DKEYWORD(package);
DKEYWORD(pragma);
DKEYWORD(private);
DKEYWORD(protected);
DKEYWORD(public);
DKEYWORD(pure);
DKEYWORD(real);
DKEYWORD(ref);
DKEYWORD(return);
DKEYWORD(scope);
DKEYWORD(shared);
DKEYWORD(short);
DKEYWORD(static);
DKEYWORD(string);
DKEYWORD(struct);
DKEYWORD(super);
DKEYWORD(switch);
DKEYWORD(synchronized);
DKEYWORD(template);
DKEYWORD(this);
DKEYWORD(throw);
DKEYWORD(true);
DKEYWORD(try);
DKEYWORD(typedef);
DKEYWORD(typeid);
DKEYWORD(typeof);
DKEYWORD(ubyte);
DKEYWORD(ucent);
DKEYWORD(uint);
DKEYWORD(ulong);
DKEYWORD(union);
DKEYWORD(unittest);
DKEYWORD(ushort);
DKEYWORD(version);
DKEYWORD(void);
DKEYWORD(volatile);
DKEYWORD(wchar);
DKEYWORD(while);
DKEYWORD(with);
DKEYWORD(wstring);

// Not really a keyword, but dispose() methods are generated in proxy classes
// and it's a special method name for D1/Tango.
DKEYWORD(dispose);

#undef DKEYWORD

#endif //D_DKW_SWG_
