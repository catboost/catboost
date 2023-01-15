/* This is a placeholder file for symbols that should be exported
 * into config_h.SH and Porting/Glossary. See also metaconfig.SH
 *
 * First version was created from the part in handy.h (which includes this)
 * H.Merijn Brand 21 Dec 2010 (Tux)
 *
 * Mentioned variables are forced to be included into config_h.SH
 * as they are only included if meta finds them referenced. That
 * implies that noone can use them unless they are available and
 * they won't be available unless used. When new symbols are probed
 * in Configure, this is the way to force them into availability.
 *
 * CHARBITS
 * GMTIME_MAX
 * GMTIME_MIN
 * HAS_ASCTIME64
 * HAS_CTIME64
 * HAS_DIFFTIME64
 * HAS_GETADDRINFO
 * HAS_GETNAMEINFO
 * HAS_GMTIME64
 * HAS_INETNTOP
 * HAS_INETPTON
 * HAS_LOCALTIME64
 * HAS_MKTIME64
 * HAS_PRCTL
 * HAS_PSEUDOFORK
 * HAS_SIN6_SCOPE_ID
 * HAS_SOCKADDR_SA_LEN
 * HAS_TIMEGM
 * I16SIZE
 * I32SIZE
 * I64SIZE
 * I8SIZE
 * LOCALTIME_MAX
 * LOCALTIME_MIN
 * LOCALTIME_R_NEEDS_TZSET
 * U16SIZE
 * U32SIZE
 * U64SIZE
 * U8SIZE
 * USE_DTRACE
 *
 */
