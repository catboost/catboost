/* ------------------------------------------------------------
 * Overloaded operator support
 * ------------------------------------------------------------ */

#ifdef __cplusplus

// operators supported in Octave, and the methods they are routed to

// __brace__      a{args}
// __brace_asgn__ a{args} = rhs
// __paren__      a(args)
// __paren_asgn__ a(args) = rhs
// __str__        generates string rep

// __not__        !a
// __uplus__      +a
// __uminus__     -a
// __transpose__  a.'
// __hermitian__  a'
// __incr__       a++
// __decr__       a--
// __add__        a + b
// __sub__        a - b
// __mul__        a * b
// __div__        a / b
// __pow__        a ^ b
// __ldiv__       a \ b
// __lt__         a < b
// __le__         a <= b
// __eq__         a == b
// __ge__         a >= b
// __gt__         a > b
// __ne__         a != b
// __el_mul__     a .* b
// __el_div__     a ./ b
// __el_pow__     a .^ b
// __el_ldiv__    a .\ b
// __el_and__     a & b
// __el_or__      a | b

// operators supported in C++, and the methods that route to them

%rename(__add__)       *::operator+;
%rename(__add__)       *::operator+();
%rename(__add__)       *::operator+() const;
%rename(__sub__)       *::operator-;
%rename(__uminus__)    *::operator-();
%rename(__uminus__)    *::operator-() const;
%rename(__mul__)       *::operator*;
%rename(__div__)       *::operator/;
%rename(__mod__)       *::operator%;
%rename(__el_and__)    *::operator&&;
%rename(__el_or__)     *::operator||;
%rename(__xor__)       *::operator^;
%rename(__invert__)    *::operator~;
%rename(__lt__)        *::operator<;
%rename(__le__)        *::operator<=;
%rename(__gt__)        *::operator>;
%rename(__ge__)        *::operator>=;
%rename(__eq__)        *::operator==;
%rename(__ne__)        *::operator!=;
%rename(__not__)       *::operator!;
%rename(__incr__)      *::operator++;
%rename(__decr__)      *::operator--;
%rename(__paren__)     *::operator();
%rename(__brace__)     *::operator[];

// Ignored inplace operators
%ignoreoperator(PLUSEQ)     operator+=;
%ignoreoperator(MINUSEQ)    operator-=;
%ignoreoperator(MULEQ)      operator*=;
%ignoreoperator(DIVEQ)      operator/=;
%ignoreoperator(MODEQ)      operator%=;
%ignoreoperator(LSHIFTEQ)   operator<<=;
%ignoreoperator(RSHIFTEQ)   operator>>=;
%ignoreoperator(ANDEQ)      operator&=;
%ignoreoperator(OREQ)       operator|=;
%ignoreoperator(XOREQ)      operator^=;

// Ignored operators
%ignoreoperator(EQ)         operator=;
%ignoreoperator(ARROWSTAR)  operator->*;
%ignoreoperator(LSHIFT)     operator<<;
%ignoreoperator(RSHIFT)     operator>>;

#endif /* __cplusplus */
