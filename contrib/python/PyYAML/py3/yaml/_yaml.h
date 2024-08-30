
#include <yaml.h>

#define PyUnicode_FromYamlString(s) PyUnicode_FromString((const char *)(void *)(s))
#define PyBytes_AS_Yaml_STRING(s) ((yaml_char_t *)PyBytes_AS_STRING(s))

#ifdef _MSC_VER	/* MS Visual C++ 6.0 */
#if _MSC_VER == 1200

#define PyLong_FromUnsignedLongLong(z)	PyInt_FromLong(i)

#endif
#endif
