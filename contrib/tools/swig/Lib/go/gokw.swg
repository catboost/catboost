/* Rename keywords.  */

#define GOKW(x) %keywordwarn("'" `x` "' is a Go keyword",rename="X%s")  `x`
#define GOBN(x) %builtinwarn("'" `x` "' conflicts with a built-in name in Go")  "::"`x`

GOKW(break);
GOKW(case);
GOKW(chan);
GOKW(const);
GOKW(continue);
GOKW(default);
GOKW(defer);
GOKW(else);
GOKW(fallthrough);
GOKW(for);
GOKW(func);
GOKW(go);
GOKW(goto);
GOKW(if);
GOKW(import);
GOKW(interface);
GOKW(package);
GOKW(range);
GOKW(return);
GOKW(select);
GOKW(struct);
GOKW(switch);
GOKW(type);
GOKW(var);

GOBN(map);

#undef GOKW
