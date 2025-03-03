/* ------------------------------------------------------------
 * Overloaded operator support
 * ------------------------------------------------------------ */

#ifdef __cplusplus

%rename(__add__)      *::operator+;
%rename(__pos__)      *::operator+();
%rename(__pos__)      *::operator+() const;
%rename(__sub__)      *::operator-;
%rename(__neg__)      *::operator-();
%rename(__neg__)      *::operator-() const;
%rename(__mul__)      *::operator*;
%rename(__div__)      *::operator/;
%rename(__mod__)      *::operator%;
%rename(__lshift__)   *::operator<<;
%rename(__rshift__)   *::operator>>;
%rename(__and__)      *::operator&;
%rename(__or__)       *::operator|;
%rename(__xor__)      *::operator^;
%rename(__invert__)   *::operator~;
%rename(__lt__)       *::operator<;
%rename(__le__)       *::operator<=;
%rename(__gt__)       *::operator>;
%rename(__ge__)       *::operator>=;
%rename(__eq__)       *::operator==;

/* Special cases */
%rename(__call__)     *::operator();

/* Ignored inplace operators */
%ignoreoperator(NOTEQUAL)   operator!=;  
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

/* Ignored operators */
%ignoreoperator(LNOT)       operator!;
%ignoreoperator(LAND)       operator&&;
%ignoreoperator(LOR)        operator||;
%ignoreoperator(EQ)         operator=;
%ignoreoperator(PLUSPLUS)   operator++;
%ignoreoperator(MINUSMINUS) operator--;
%ignoreoperator(ARROWSTAR)  operator->*;
%ignoreoperator(INDEX)      operator[];


#endif /* __cplusplus */
