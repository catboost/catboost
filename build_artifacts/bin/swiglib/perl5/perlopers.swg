/* ------------------------------------------------------------
 * Overloaded operator support
 * ------------------------------------------------------------ */

#ifdef __cplusplus

// These are auto-supported by the Perl-module
%rename(__plusplus__) *::operator++;
%rename(__minmin__)   *::operator--;
%rename(__add__)      *::operator+;
%rename(__sub__)      *::operator-;
%rename(__neg__)      *::operator-();
%rename(__neg__)      *::operator-() const;
%rename(__mul__)      *::operator*;
%rename(__div__)      *::operator/;
%rename(__eq__)       *::operator==;
%rename(__ne__)       *::operator!=;
%rename(__mod__)      *::operator%;
%rename(__gt__)       *::operator>;
%rename(__lt__)       *::operator<;
%rename(__not__)      *::operator!;
%rename(__le__)       *::operator<=;
%rename(__ge__)       *::operator>=;
%rename(__and__)      *::operator&;
%rename(__or__)       *::operator|;
%rename(__iadd__)     *::operator+=;  
%rename(__isub__)     *::operator-=;  

// These are renamed, but no test exists in operator_overload_runme.pl
%ignoreoperator(EQ)         operator=;

// These are renamed, but no 'use overload...' is added
%rename(__lshift__)   *::operator<<;
%rename(__rshift__)   *::operator>>;
%rename(__xor__)      *::operator^;
%rename(__invert__)   *::operator~;
%rename(__call__)     *::operator();

/* Ignored operators */
%ignoreoperator(LAND)       operator&&;
%ignoreoperator(LOR)        operator||;
%ignoreoperator(MULEQ)      operator*=;  
%ignoreoperator(DIVEQ)      operator/=;  
%ignoreoperator(MODEQ)      operator%=;  
%ignoreoperator(LSHIFTEQ)   operator<<=; 
%ignoreoperator(RSHIFTEQ)   operator>>=; 
%ignoreoperator(ANDEQ)      operator&=;  
%ignoreoperator(OREQ)       operator|=;  
%ignoreoperator(XOREQ)      operator^=;  
%ignoreoperator(ARROWSTAR)  operator->*;
%ignoreoperator(INDEX)      operator[];


#endif /* __cplusplus */
