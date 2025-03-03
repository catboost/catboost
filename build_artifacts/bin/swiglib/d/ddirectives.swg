/* -----------------------------------------------------------------------------
 * ddirectives.swg
 *
 * D-specifiv directives.
 * ----------------------------------------------------------------------------- */

#define %dmanifestconst             %feature("d:manifestconst")
#define %dconstvalue(value)         %feature("d:constvalue",value)
#define %dmethodmodifiers           %feature("d:methodmodifiers")
#define %dnothrowexception          %feature("except")
#define %proxycode                  %insert("proxycode")
