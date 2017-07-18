#pragma once

#ifdef __cplusplus
extern "C" {
#endif

struct plain32_base64_state {
	int eof;
	int bytes;
	unsigned char carry;
};

/* Wrapper function to encode a plain string of given length. Output is written
 * to *out without trailing zero. Output length in bytes is written to *outlen.
 * The buffer in `out` has been allocated by the caller and is at least 4/3 the
 * size of the input. See above for `flags`; set to 0 for default operation: */
void plain32_base64_encode
	( const char		*src
	, size_t		 srclen
	, char			*out
	, size_t		*outlen
	) ;

/* Call this before calling base64_stream_encode() to init the state. See above
 * for `flags`; set to 0 for default operation: */
void plain32_base64_stream_encode_init
	( struct plain32_base64_state	*state
	) ;

/* Encodes the block of data of given length at `src`, into the buffer at
 * `out`. Caller is responsible for allocating a large enough out-buffer; it
 * must be at least 4/3 the size of the in-buffer, but take some margin. Places
 * the number of new bytes written into `outlen` (which is set to zero when the
 * function starts). Does not zero-terminate or finalize the output. */
void plain32_base64_stream_encode
	( struct plain32_base64_state	*state
	, const char		*src
	, size_t		 srclen
	, char			*out
	, size_t		*outlen
	) ;

/* Finalizes the output begun by previous calls to `base64_stream_encode()`.
 * Adds the required end-of-stream markers if appropriate. `outlen` is modified
 * and will contain the number of new bytes written at `out` (which will quite
 * often be zero). */
void plain32_base64_stream_encode_final
	( struct plain32_base64_state	*state
	, char			*out
	, size_t		*outlen
	) ;

/* Wrapper function to decode a plain string of given length. Output is written
 * to *out without trailing zero. Output length in bytes is written to *outlen.
 * The buffer in `out` has been allocated by the caller and is at least 3/4 the
 * size of the input. See above for `flags`, set to 0 for default operation: */
int plain32_base64_decode
	( const char		*src
	, size_t		 srclen
	, char			*out
	, size_t		*outlen
	) ;

/* Call this before calling base64_stream_decode() to init the state. See above
 * for `flags`; set to 0 for default operation: */
void plain32_base64_stream_decode_init
	( struct plain32_base64_state	*state
	) ;

/* Decodes the block of data of given length at `src`, into the buffer at
 * `out`. Caller is responsible for allocating a large enough out-buffer; it
 * must be at least 3/4 the size of the in-buffer, but take some margin. Places
 * the number of new bytes written into `outlen` (which is set to zero when the
 * function starts). Does not zero-terminate the output. Returns 1 if all is
 * well, and 0 if a decoding error was found, such as an invalid character.
 * Returns -1 if the chosen codec is not included in the current build. Used by
 * the test harness to check whether a codec is available for testing. */
int plain32_base64_stream_decode
	( struct plain32_base64_state	*state
	, const char		*src
	, size_t		 srclen
	, char			*out
	, size_t		*outlen
	) ;

#ifdef __cplusplus
}
#endif

