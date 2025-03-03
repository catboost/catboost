/*
  Warnings for Lua keywords, built-in names and bad names.
*/

#define LUAKW(x) %keywordwarn("'" `x` "' is a Lua keyword, renaming to 'c_" `x` "'", rename="c_%s")  `x`
#define LUABN(x) %namewarn(%warningmsg(SWIGWARN_PARSE_BUILTIN_NAME, "'" `x` "' conflicts with a basic function in Lua"), %$not %$ismember)  `x`

/*
  Warnings for Lua keywords 
  http://www.lua.org/manual/5.2/manual.html#3.1
*/

LUAKW(and);
LUAKW(break);
LUAKW(do);
LUAKW(else);
LUAKW(elseif);
LUAKW(end);
LUAKW(false);
LUAKW(for);
LUAKW(function);
LUAKW(goto);
LUAKW(if);
LUAKW(in);
LUAKW(local);
LUAKW(nil);
LUAKW(not);
LUAKW(or);
LUAKW(repeat);
LUAKW(return);
LUAKW(then);
LUAKW(true);
LUAKW(until);
LUAKW(while);

/*
  Basic functions
  http://www.lua.org/manual/5.2/manual.html#6.1
*/ 

LUABN(assert);
LUABN(collectgarbage);
LUABN(dofile);
LUABN(error);
LUABN(_G); // Not actually a function
LUABN(getmetatable);
LUABN(ipairs);
LUABN(load);
LUABN(loadfile);
LUABN(next);
LUABN(pairs);
LUABN(pcall);
LUABN(print);
LUABN(rawequal);
LUABN(rawget);
LUABN(rawlen);
LUABN(rawset);
LUABN(select);
LUABN(setmetatable);
LUABN(tonumber);
LUABN(tostring);
LUABN(type);
LUABN(_VERSION); // Not actually a function
LUABN(xpcall);

#undef LUABN
#undef LUAKW
