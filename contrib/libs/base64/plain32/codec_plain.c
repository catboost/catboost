#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>

#include "libbase64.h"
#include "codecs.h"

void
plain32_base64_stream_encode
	( struct plain32_base64_state	*state
	, const char		*src
	, size_t		 srclen
	, char			*out
	, size_t		*outlen
	)
{
	#include "enc_head.c"
	#include "enc_uint32.c"
	#include "enc_tail.c"
}

int
plain32_base64_stream_decode
	( struct plain32_base64_state	*state
	, const char		*src
	, size_t		 srclen
	, char			*out
	, size_t		*outlen
	)
{
	#include "dec_head.c"
	#include "dec_uint32.c"
	#include "dec_tail.c"
}
