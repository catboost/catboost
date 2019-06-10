/*

Copyright (c) 2012, Lambda Foundry, Inc., except where noted

Incorporates components of WarrenWeckesser/textreader, licensed under 3-clause
BSD

See LICENSE for the license

*/

/*

Low-level ascii-file processing for pandas. Combines some elements from
Python's built-in csv module and Warren Weckesser's textreader project on
GitHub. See Python Software Foundation License and BSD licenses for these.

*/

#include "tokenizer.h"

#include <ctype.h>
#include <float.h>
#include <math.h>

#include "../headers/portable.h"

static void *safe_realloc(void *buffer, size_t size) {
    void *result;
    // OSX is weird.
    // http://stackoverflow.com/questions/9560609/
    // different-realloc-behaviour-in-linux-and-osx

    result = realloc(buffer, size);
    TRACE(("safe_realloc: buffer = %p, size = %zu, result = %p\n", buffer, size,
           result))

    return result;
}

void coliter_setup(coliter_t *self, parser_t *parser, int i, int start) {
    // column i, starting at 0
    self->words = parser->words;
    self->col = i;
    self->line_start = parser->line_start + start;
}

coliter_t *coliter_new(parser_t *self, int i) {
    // column i, starting at 0
    coliter_t *iter = (coliter_t *)malloc(sizeof(coliter_t));

    if (NULL == iter) {
        return NULL;
    }

    coliter_setup(iter, self, i, 0);
    return iter;
}

static void free_if_not_null(void **ptr) {
    TRACE(("free_if_not_null %p\n", *ptr))
    if (*ptr != NULL) {
        free(*ptr);
        *ptr = NULL;
    }
}

/*

  Parser / tokenizer

*/

static void *grow_buffer(void *buffer, int64_t length, int64_t *capacity,
                         int64_t space, int64_t elsize, int *error) {
    int64_t cap = *capacity;
    void *newbuffer = buffer;

    // Can we fit potentially nbytes tokens (+ null terminators) in the stream?
    while ((length + space >= cap) && (newbuffer != NULL)) {
        cap = cap ? cap << 1 : 2;
        buffer = newbuffer;
        newbuffer = safe_realloc(newbuffer, elsize * cap);
    }

    if (newbuffer == NULL) {
        // realloc failed so don't change *capacity, set *error to errno
        // and return the last good realloc'd buffer so it can be freed
        *error = errno;
        newbuffer = buffer;
    } else {
        // realloc worked, update *capacity and set *error to 0
        // sigh, multiple return values
        *capacity = cap;
        *error = 0;
    }
    return newbuffer;
}

void parser_set_default_options(parser_t *self) {
    self->decimal = '.';
    self->sci = 'E';

    // For tokenization
    self->state = START_RECORD;

    self->delimiter = ',';  // XXX
    self->delim_whitespace = 0;

    self->doublequote = 0;
    self->quotechar = '"';
    self->escapechar = 0;

    self->lineterminator = '\0'; /* NUL->standard logic */

    self->skipinitialspace = 0;
    self->quoting = QUOTE_MINIMAL;
    self->allow_embedded_newline = 1;
    self->strict = 0;

    self->expected_fields = -1;
    self->error_bad_lines = 0;
    self->warn_bad_lines = 0;

    self->commentchar = '#';
    self->thousands = '\0';

    self->skipset = NULL;
    self->skipfunc = NULL;
    self->skip_first_N_rows = -1;
    self->skip_footer = 0;
}

int get_parser_memory_footprint(parser_t *self) { return 0; }

parser_t *parser_new() { return (parser_t *)calloc(1, sizeof(parser_t)); }

int parser_clear_data_buffers(parser_t *self) {
    free_if_not_null((void *)&self->stream);
    free_if_not_null((void *)&self->words);
    free_if_not_null((void *)&self->word_starts);
    free_if_not_null((void *)&self->line_start);
    free_if_not_null((void *)&self->line_fields);
    return 0;
}

int parser_cleanup(parser_t *self) {
    int status = 0;

    // XXX where to put this
    free_if_not_null((void *)&self->error_msg);
    free_if_not_null((void *)&self->warn_msg);

    if (self->skipset != NULL) {
        kh_destroy_int64((kh_int64_t *)self->skipset);
        self->skipset = NULL;
    }

    if (parser_clear_data_buffers(self) < 0) {
        status = -1;
    }

    if (self->cb_cleanup != NULL) {
        if (self->cb_cleanup(self->source) < 0) {
            status = -1;
        }
        self->cb_cleanup = NULL;
    }

    return status;
}

int parser_init(parser_t *self) {
    int64_t sz;

    /*
      Initialize data buffers
    */

    self->stream = NULL;
    self->words = NULL;
    self->word_starts = NULL;
    self->line_start = NULL;
    self->line_fields = NULL;
    self->error_msg = NULL;
    self->warn_msg = NULL;

    // token stream
    self->stream = (char *)malloc(STREAM_INIT_SIZE * sizeof(char));
    if (self->stream == NULL) {
        parser_cleanup(self);
        return PARSER_OUT_OF_MEMORY;
    }
    self->stream_cap = STREAM_INIT_SIZE;
    self->stream_len = 0;

    // word pointers and metadata
    sz = STREAM_INIT_SIZE / 10;
    sz = sz ? sz : 1;
    self->words = (char **)malloc(sz * sizeof(char *));
    self->word_starts = (int64_t *)malloc(sz * sizeof(int64_t));
    self->max_words_cap = sz;
    self->words_cap = sz;
    self->words_len = 0;

    // line pointers and metadata
    self->line_start = (int64_t *)malloc(sz * sizeof(int64_t));

    self->line_fields = (int64_t *)malloc(sz * sizeof(int64_t));

    self->lines_cap = sz;
    self->lines = 0;
    self->file_lines = 0;

    if (self->stream == NULL || self->words == NULL ||
        self->word_starts == NULL || self->line_start == NULL ||
        self->line_fields == NULL) {
        parser_cleanup(self);

        return PARSER_OUT_OF_MEMORY;
    }

    /* amount of bytes buffered */
    self->datalen = 0;
    self->datapos = 0;

    self->line_start[0] = 0;
    self->line_fields[0] = 0;

    self->pword_start = self->stream;
    self->word_start = 0;

    self->state = START_RECORD;

    self->error_msg = NULL;
    self->warn_msg = NULL;

    self->commentchar = '\0';

    return 0;
}

void parser_free(parser_t *self) {
    // opposite of parser_init
    parser_cleanup(self);
}

void parser_del(parser_t *self) {
    free(self);
}

static int make_stream_space(parser_t *self, size_t nbytes) {
    int64_t i, cap, length;
    int status;
    void *orig_ptr, *newptr;

    // Can we fit potentially nbytes tokens (+ null terminators) in the stream?

    /*
      TOKEN STREAM
    */

    orig_ptr = (void *)self->stream;
    TRACE(
        ("\n\nmake_stream_space: nbytes = %zu.  grow_buffer(self->stream...)\n",
         nbytes))
    self->stream = (char *)grow_buffer((void *)self->stream, self->stream_len,
                                       (int64_t*)&self->stream_cap, nbytes * 2,
                                       sizeof(char), &status);
    TRACE(
        ("make_stream_space: self->stream=%p, self->stream_len = %zu, "
         "self->stream_cap=%zu, status=%zu\n",
         self->stream, self->stream_len, self->stream_cap, status))

    if (status != 0) {
        return PARSER_OUT_OF_MEMORY;
    }

    // realloc sets errno when moving buffer?
    if (self->stream != orig_ptr) {
        self->pword_start = self->stream + self->word_start;

        for (i = 0; i < self->words_len; ++i) {
            self->words[i] = self->stream + self->word_starts[i];
        }
    }

    /*
      WORD VECTORS
    */

    cap = self->words_cap;

    /**
     * If we are reading in chunks, we need to be aware of the maximum number
     * of words we have seen in previous chunks (self->max_words_cap), so
     * that way, we can properly allocate when reading subsequent ones.
     *
     * Otherwise, we risk a buffer overflow if we mistakenly under-allocate
     * just because a recent chunk did not have as many words.
     */
    if (self->words_len + nbytes < self->max_words_cap) {
        length = self->max_words_cap - nbytes - 1;
    } else {
        length = self->words_len;
    }

    self->words =
        (char **)grow_buffer((void *)self->words, length,
                             (int64_t*)&self->words_cap, nbytes,
                             sizeof(char *), &status);
    TRACE(
        ("make_stream_space: grow_buffer(self->self->words, %zu, %zu, %zu, "
         "%d)\n",
         self->words_len, self->words_cap, nbytes, status))
    if (status != 0) {
        return PARSER_OUT_OF_MEMORY;
    }

    // realloc took place
    if (cap != self->words_cap) {
        TRACE(
            ("make_stream_space: cap != self->words_cap, nbytes = %d, "
             "self->words_cap=%d\n",
             nbytes, self->words_cap))
        newptr = safe_realloc((void *)self->word_starts,
                              sizeof(int64_t) * self->words_cap);
        if (newptr == NULL) {
            return PARSER_OUT_OF_MEMORY;
        } else {
            self->word_starts = (int64_t *)newptr;
        }
    }

    /*
      LINE VECTORS
    */
    cap = self->lines_cap;
    self->line_start =
        (int64_t *)grow_buffer((void *)self->line_start, self->lines + 1,
                           (int64_t*)&self->lines_cap, nbytes,
                           sizeof(int64_t), &status);
    TRACE((
        "make_stream_space: grow_buffer(self->line_start, %zu, %zu, %zu, %d)\n",
        self->lines + 1, self->lines_cap, nbytes, status))
    if (status != 0) {
        return PARSER_OUT_OF_MEMORY;
    }

    // realloc took place
    if (cap != self->lines_cap) {
        TRACE(("make_stream_space: cap != self->lines_cap, nbytes = %d\n",
               nbytes))
        newptr = safe_realloc((void *)self->line_fields,
                              sizeof(int64_t) * self->lines_cap);
        if (newptr == NULL) {
            return PARSER_OUT_OF_MEMORY;
        } else {
            self->line_fields = (int64_t *)newptr;
        }
    }

    return 0;
}

static int push_char(parser_t *self, char c) {
    TRACE(("push_char: self->stream[%zu] = %x, stream_cap=%zu\n",
           self->stream_len + 1, c, self->stream_cap))
    if (self->stream_len >= self->stream_cap) {
        TRACE(
            ("push_char: ERROR!!! self->stream_len(%d) >= "
             "self->stream_cap(%d)\n",
             self->stream_len, self->stream_cap))
        int64_t bufsize = 100;
        self->error_msg = (char *)malloc(bufsize);
        snprintf(self->error_msg, bufsize,
                 "Buffer overflow caught - possible malformed input file.\n");
        return PARSER_OUT_OF_MEMORY;
    }
    self->stream[self->stream_len++] = c;
    return 0;
}

int PANDAS_INLINE end_field(parser_t *self) {
    // XXX cruft
    if (self->words_len >= self->words_cap) {
        TRACE(
            ("end_field: ERROR!!! self->words_len(%zu) >= "
             "self->words_cap(%zu)\n",
             self->words_len, self->words_cap))
        int64_t bufsize = 100;
        self->error_msg = (char *)malloc(bufsize);
        snprintf(self->error_msg, bufsize,
                 "Buffer overflow caught - possible malformed input file.\n");
        return PARSER_OUT_OF_MEMORY;
    }

    // null terminate token
    push_char(self, '\0');

    // set pointer and metadata
    self->words[self->words_len] = self->pword_start;

    TRACE(("end_field: Char diff: %d\n", self->pword_start - self->words[0]));

    TRACE(("end_field: Saw word %s at: %d. Total: %d\n", self->pword_start,
           self->word_start, self->words_len + 1))

    self->word_starts[self->words_len] = self->word_start;
    self->words_len++;

    // increment line field count
    self->line_fields[self->lines]++;

    // New field begin in stream
    self->pword_start = self->stream + self->stream_len;
    self->word_start = self->stream_len;

    return 0;
}

static void append_warning(parser_t *self, const char *msg) {
    int64_t ex_length;
    int64_t length = strlen(msg);
    void *newptr;

    if (self->warn_msg == NULL) {
        self->warn_msg = (char *)malloc(length + 1);
        strncpy(self->warn_msg, msg, strlen(msg) + 1);
    } else {
        ex_length = strlen(self->warn_msg);
        newptr = safe_realloc(self->warn_msg, ex_length + length + 1);
        if (newptr != NULL) {
            self->warn_msg = (char *)newptr;
            strncpy(self->warn_msg + ex_length, msg, strlen(msg) + 1);
        }
    }
}

static int end_line(parser_t *self) {
    char *msg;
    int64_t fields;
    int ex_fields = self->expected_fields;
    int64_t bufsize = 100;  // for error or warning messages

    fields = self->line_fields[self->lines];

    TRACE(("end_line: Line end, nfields: %d\n", fields));

    TRACE(("end_line: lines: %d\n", self->lines));
    if (self->lines > 0) {
        if (self->expected_fields >= 0) {
            ex_fields = self->expected_fields;
        } else {
            ex_fields = self->line_fields[self->lines - 1];
        }
    }
    TRACE(("end_line: ex_fields: %d\n", ex_fields));

    if (self->state == START_FIELD_IN_SKIP_LINE ||
        self->state == IN_FIELD_IN_SKIP_LINE ||
        self->state == IN_QUOTED_FIELD_IN_SKIP_LINE ||
        self->state == QUOTE_IN_QUOTED_FIELD_IN_SKIP_LINE) {
        TRACE(("end_line: Skipping row %d\n", self->file_lines));
        // increment file line count
        self->file_lines++;

        // skip the tokens from this bad line
        self->line_start[self->lines] += fields;

        // reset field count
        self->line_fields[self->lines] = 0;
        return 0;
    }

    if (!(self->lines <= (int64_t) self->header_end + 1) &&
        (self->expected_fields < 0 && fields > ex_fields) && !(self->usecols)) {
        // increment file line count
        self->file_lines++;

        // skip the tokens from this bad line
        self->line_start[self->lines] += fields;

        // reset field count
        self->line_fields[self->lines] = 0;

        // file_lines is now the actual file line number (starting at 1)
        if (self->error_bad_lines) {
            self->error_msg = (char *)malloc(bufsize);
            snprintf(self->error_msg, bufsize,
                    "Expected %d fields in line %lld, saw %lld\n",
                    ex_fields, (long long)self->file_lines, (long long)fields);

            TRACE(("Error at line %d, %d fields\n", self->file_lines, fields));

            return -1;
        } else {
            // simply skip bad lines
            if (self->warn_bad_lines) {
                // pass up error message
                msg = (char *)malloc(bufsize);
                snprintf(msg, bufsize,
                        "Skipping line %lld: expected %d fields, saw %lld\n",
                         (long long)self->file_lines, ex_fields,
                         (long long)fields);
                append_warning(self, msg);
                free(msg);
            }
        }
    } else {
        // missing trailing delimiters
        if ((self->lines >= (int64_t) self->header_end + 1) &&
                fields < ex_fields) {
            // might overrun the buffer when closing fields
            if (make_stream_space(self, ex_fields - fields) < 0) {
                int64_t bufsize = 100;
                self->error_msg = (char *)malloc(bufsize);
                snprintf(self->error_msg, bufsize, "out of memory");
                return -1;
            }

            while (fields < ex_fields) {
                end_field(self);
                fields++;
            }
        }

        // increment both line counts
        self->file_lines++;
        self->lines++;

        // good line, set new start point
        if (self->lines >= self->lines_cap) {
            TRACE((
                "end_line: ERROR!!! self->lines(%zu) >= self->lines_cap(%zu)\n",
                self->lines, self->lines_cap))
            int64_t bufsize = 100;
            self->error_msg = (char *)malloc(bufsize);
            snprintf(self->error_msg, bufsize,
                     "Buffer overflow caught - "
                     "possible malformed input file.\n");
            return PARSER_OUT_OF_MEMORY;
        }
        self->line_start[self->lines] =
            (self->line_start[self->lines - 1] + fields);

        TRACE(
            ("end_line: new line start: %d\n", self->line_start[self->lines]));

        // new line start with 0 fields
        self->line_fields[self->lines] = 0;
    }

    TRACE(("end_line: Finished line, at %d\n", self->lines));

    return 0;
}

int parser_add_skiprow(parser_t *self, int64_t row) {
    khiter_t k;
    kh_int64_t *set;
    int ret = 0;

    if (self->skipset == NULL) {
        self->skipset = (void *)kh_init_int64();
    }

    set = (kh_int64_t *)self->skipset;

    k = kh_put_int64(set, row, &ret);
    set->keys[k] = row;

    return 0;
}

int parser_set_skipfirstnrows(parser_t *self, int64_t nrows) {
    // self->file_lines is zero based so subtract 1 from nrows
    if (nrows > 0) {
        self->skip_first_N_rows = nrows - 1;
    }

    return 0;
}

static int parser_buffer_bytes(parser_t *self, size_t nbytes) {
    int status;
    size_t bytes_read;

    status = 0;
    self->datapos = 0;
    self->data = self->cb_io(self->source, nbytes, &bytes_read, &status);
    TRACE((
        "parser_buffer_bytes self->cb_io: nbytes=%zu, datalen: %d, status=%d\n",
        nbytes, bytes_read, status));
    self->datalen = bytes_read;

    if (status != REACHED_EOF && self->data == NULL) {
        int64_t bufsize = 200;
        self->error_msg = (char *)malloc(bufsize);

        if (status == CALLING_READ_FAILED) {
            snprintf(self->error_msg, bufsize,
                     "Calling read(nbytes) on source failed. "
                     "Try engine='python'.");
        } else {
            snprintf(self->error_msg, bufsize, "Unknown error in IO callback");
        }
        return -1;
    }

    TRACE(("datalen: %d\n", self->datalen));

    return status;
}

/*

  Tokenization macros and state machine code

*/

#define PUSH_CHAR(c)                                                          \
    TRACE(                                                                    \
        ("PUSH_CHAR: Pushing %c, slen= %d, stream_cap=%zu, stream_len=%zu\n", \
         c, slen, self->stream_cap, self->stream_len))                        \
    if (slen >= self->stream_cap) {                                           \
        TRACE(("PUSH_CHAR: ERROR!!! slen(%d) >= stream_cap(%d)\n", slen,      \
               self->stream_cap))                                             \
        int64_t bufsize = 100;                                                \
        self->error_msg = (char *)malloc(bufsize);                            \
        snprintf(self->error_msg, bufsize,                                    \
                 "Buffer overflow caught - possible malformed input file.\n");\
        return PARSER_OUT_OF_MEMORY;                                          \
    }                                                                         \
    *stream++ = c;                                                            \
    slen++;

// This is a little bit of a hack but works for now

#define END_FIELD()                           \
    self->stream_len = slen;                  \
    if (end_field(self) < 0) {                \
        goto parsingerror;                    \
    }                                         \
    stream = self->stream + self->stream_len; \
    slen = self->stream_len;

#define END_LINE_STATE(STATE)                                        \
    self->stream_len = slen;                                         \
    if (end_line(self) < 0) {                                        \
        goto parsingerror;                                           \
    }                                                                \
    stream = self->stream + self->stream_len;                        \
    slen = self->stream_len;                                         \
    self->state = STATE;                                             \
    if (line_limit > 0 && self->lines == start_lines + (int64_t)line_limit) {  \
        goto linelimit;                                              \
    }

#define END_LINE_AND_FIELD_STATE(STATE)                              \
    self->stream_len = slen;                                         \
    if (end_line(self) < 0) {                                        \
        goto parsingerror;                                           \
    }                                                                \
    if (end_field(self) < 0) {                                       \
        goto parsingerror;                                           \
    }                                                                \
    stream = self->stream + self->stream_len;                        \
    slen = self->stream_len;                                         \
    self->state = STATE;                                             \
    if (line_limit > 0 && self->lines == start_lines + (int64_t)line_limit) { \
        goto linelimit;                                              \
    }

#define END_LINE() END_LINE_STATE(START_RECORD)

#define IS_WHITESPACE(c) ((c == ' ' || c == '\t'))

#define IS_TERMINATOR(c)                            \
    ((self->lineterminator == '\0' && c == '\n') || \
     (self->lineterminator != '\0' && c == self->lineterminator))

#define IS_QUOTE(c) ((c == self->quotechar && self->quoting != QUOTE_NONE))

// don't parse '\r' with a custom line terminator
#define IS_CARRIAGE(c) ((self->lineterminator == '\0' && c == '\r'))

#define IS_COMMENT_CHAR(c) \
    ((self->commentchar != '\0' && c == self->commentchar))

#define IS_ESCAPE_CHAR(c) ((self->escapechar != '\0' && c == self->escapechar))

#define IS_SKIPPABLE_SPACE(c) \
    ((!self->delim_whitespace && c == ' ' && self->skipinitialspace))

// applied when in a field
#define IS_DELIMITER(c)                                   \
    ((!self->delim_whitespace && c == self->delimiter) || \
     (self->delim_whitespace && IS_WHITESPACE(c)))

#define _TOKEN_CLEANUP()                                                \
    self->stream_len = slen;                                            \
    self->datapos = i;                                                  \
    TRACE(("_TOKEN_CLEANUP: datapos: %d, datalen: %d\n", self->datapos, \
           self->datalen));

#define CHECK_FOR_BOM()                                                   \
    if (*buf == '\xef' && *(buf + 1) == '\xbb' && *(buf + 2) == '\xbf') { \
        buf += 3;                                                         \
        self->datapos += 3;                                               \
    }

int skip_this_line(parser_t *self, int64_t rownum) {
    int should_skip;
    PyObject *result;
    PyGILState_STATE state;

    if (self->skipfunc != NULL) {
        state = PyGILState_Ensure();
        result = PyObject_CallFunction(self->skipfunc, "i", rownum);

        // Error occurred. It will be processed
        // and caught at the Cython level.
        if (result == NULL) {
            should_skip = -1;
        } else {
            should_skip = PyObject_IsTrue(result);
        }

        Py_XDECREF(result);
        PyGILState_Release(state);

        return should_skip;
    } else if (self->skipset != NULL) {
        return (kh_get_int64((kh_int64_t *)self->skipset, self->file_lines) !=
                ((kh_int64_t *)self->skipset)->n_buckets);
    } else {
        return (rownum <= self->skip_first_N_rows);
    }
}

int tokenize_bytes(parser_t *self, size_t line_limit, int64_t start_lines) {
    int64_t i, slen;
    int should_skip;
    char c;
    char *stream;
    char *buf = self->data + self->datapos;

    if (make_stream_space(self, self->datalen - self->datapos) < 0) {
        int64_t bufsize = 100;
        self->error_msg = (char *)malloc(bufsize);
        snprintf(self->error_msg, bufsize, "out of memory");
        return -1;
    }

    stream = self->stream + self->stream_len;
    slen = self->stream_len;

    TRACE(("%s\n", buf));

    if (self->file_lines == 0) {
        CHECK_FOR_BOM();
    }

    for (i = self->datapos; i < self->datalen; ++i) {
        // next character in file
        c = *buf++;

        TRACE(
            ("tokenize_bytes - Iter: %d Char: 0x%x Line %d field_count %d, "
             "state %d\n",
             i, c, self->file_lines + 1, self->line_fields[self->lines],
             self->state));

        switch (self->state) {
            case START_FIELD_IN_SKIP_LINE:
                if (IS_TERMINATOR(c)) {
                    END_LINE();
                } else if (IS_CARRIAGE(c)) {
                    self->file_lines++;
                    self->state = EAT_CRNL_NOP;
                } else if (IS_QUOTE(c)) {
                    self->state = IN_QUOTED_FIELD_IN_SKIP_LINE;
                } else if (IS_DELIMITER(c)) {
                    // Do nothing, we're starting a new field again.
                } else {
                    self->state = IN_FIELD_IN_SKIP_LINE;
                }
                break;

            case IN_FIELD_IN_SKIP_LINE:
                if (IS_TERMINATOR(c)) {
                    END_LINE();
                } else if (IS_CARRIAGE(c)) {
                    self->file_lines++;
                    self->state = EAT_CRNL_NOP;
                } else if (IS_DELIMITER(c)) {
                    self->state = START_FIELD_IN_SKIP_LINE;
                }
                break;

            case IN_QUOTED_FIELD_IN_SKIP_LINE:
                if (IS_QUOTE(c)) {
                    if (self->doublequote) {
                        self->state = QUOTE_IN_QUOTED_FIELD_IN_SKIP_LINE;
                    } else {
                        self->state = IN_FIELD_IN_SKIP_LINE;
                    }
                }
                break;

            case QUOTE_IN_QUOTED_FIELD_IN_SKIP_LINE:
                if (IS_QUOTE(c)) {
                    self->state = IN_QUOTED_FIELD_IN_SKIP_LINE;
                } else if (IS_TERMINATOR(c)) {
                    END_LINE();
                } else if (IS_CARRIAGE(c)) {
                    self->file_lines++;
                    self->state = EAT_CRNL_NOP;
                } else if (IS_DELIMITER(c)) {
                    self->state = START_FIELD_IN_SKIP_LINE;
                } else {
                    self->state = IN_FIELD_IN_SKIP_LINE;
                }
                break;

            case WHITESPACE_LINE:
                if (IS_TERMINATOR(c)) {
                    self->file_lines++;
                    self->state = START_RECORD;
                    break;
                } else if (IS_CARRIAGE(c)) {
                    self->file_lines++;
                    self->state = EAT_CRNL_NOP;
                    break;
                } else if (!self->delim_whitespace) {
                    if (IS_WHITESPACE(c) && c != self->delimiter) {
                    } else {  // backtrack
                        // use i + 1 because buf has been incremented but not i
                        do {
                            --buf;
                            --i;
                        } while (i + 1 > self->datapos && !IS_TERMINATOR(*buf));

                        // reached a newline rather than the beginning
                        if (IS_TERMINATOR(*buf)) {
                            ++buf;  // move pointer to first char after newline
                            ++i;
                        }
                        self->state = START_FIELD;
                    }
                    break;
                }
            // fall through

            case EAT_WHITESPACE:
                if (IS_TERMINATOR(c)) {
                    END_LINE();
                    self->state = START_RECORD;
                    break;
                } else if (IS_CARRIAGE(c)) {
                    self->state = EAT_CRNL;
                    break;
                } else if (IS_COMMENT_CHAR(c)) {
                    self->state = EAT_COMMENT;
                    break;
                } else if (!IS_WHITESPACE(c)) {
                    self->state = START_FIELD;
                    // fall through to subsequent state
                } else {
                    // if whitespace char, keep slurping
                    break;
                }

            case START_RECORD:
                // start of record
                should_skip = skip_this_line(self, self->file_lines);

                if (should_skip == -1) {
                    goto parsingerror;
                } else if (should_skip) {
                    if (IS_QUOTE(c)) {
                        self->state = IN_QUOTED_FIELD_IN_SKIP_LINE;
                    } else {
                        self->state = IN_FIELD_IN_SKIP_LINE;

                        if (IS_TERMINATOR(c)) {
                            END_LINE();
                        }
                    }
                    break;
                } else if (IS_TERMINATOR(c)) {
                    // \n\r possible?
                    if (self->skip_empty_lines) {
                        self->file_lines++;
                    } else {
                        END_LINE();
                    }
                    break;
                } else if (IS_CARRIAGE(c)) {
                    if (self->skip_empty_lines) {
                        self->file_lines++;
                        self->state = EAT_CRNL_NOP;
                    } else {
                        self->state = EAT_CRNL;
                    }
                    break;
                } else if (IS_COMMENT_CHAR(c)) {
                    self->state = EAT_LINE_COMMENT;
                    break;
                } else if (IS_WHITESPACE(c)) {
                    if (self->delim_whitespace) {
                        if (self->skip_empty_lines) {
                            self->state = WHITESPACE_LINE;
                        } else {
                            self->state = EAT_WHITESPACE;
                        }
                        break;
                    } else if (c != self->delimiter && self->skip_empty_lines) {
                        self->state = WHITESPACE_LINE;
                        break;
                    }
                    // fall through
                }

                // normal character - fall through
                // to handle as START_FIELD
                self->state = START_FIELD;

            case START_FIELD:
                // expecting field
                if (IS_TERMINATOR(c)) {
                    END_FIELD();
                    END_LINE();
                } else if (IS_CARRIAGE(c)) {
                    END_FIELD();
                    self->state = EAT_CRNL;
                } else if (IS_QUOTE(c)) {
                    // start quoted field
                    self->state = IN_QUOTED_FIELD;
                } else if (IS_ESCAPE_CHAR(c)) {
                    // possible escaped character
                    self->state = ESCAPED_CHAR;
                } else if (IS_SKIPPABLE_SPACE(c)) {
                    // ignore space at start of field
                } else if (IS_DELIMITER(c)) {
                    if (self->delim_whitespace) {
                        self->state = EAT_WHITESPACE;
                    } else {
                        // save empty field
                        END_FIELD();
                    }
                } else if (IS_COMMENT_CHAR(c)) {
                    END_FIELD();
                    self->state = EAT_COMMENT;
                } else {
                    // begin new unquoted field
                    PUSH_CHAR(c);
                    self->state = IN_FIELD;
                }
                break;

            case ESCAPED_CHAR:
                PUSH_CHAR(c);
                self->state = IN_FIELD;
                break;

            case EAT_LINE_COMMENT:
                if (IS_TERMINATOR(c)) {
                    self->file_lines++;
                    self->state = START_RECORD;
                } else if (IS_CARRIAGE(c)) {
                    self->file_lines++;
                    self->state = EAT_CRNL_NOP;
                }
                break;

            case IN_FIELD:
                // in unquoted field
                if (IS_TERMINATOR(c)) {
                    END_FIELD();
                    END_LINE();
                } else if (IS_CARRIAGE(c)) {
                    END_FIELD();
                    self->state = EAT_CRNL;
                } else if (IS_ESCAPE_CHAR(c)) {
                    // possible escaped character
                    self->state = ESCAPED_CHAR;
                } else if (IS_DELIMITER(c)) {
                    // end of field - end of line not reached yet
                    END_FIELD();

                    if (self->delim_whitespace) {
                        self->state = EAT_WHITESPACE;
                    } else {
                        self->state = START_FIELD;
                    }
                } else if (IS_COMMENT_CHAR(c)) {
                    END_FIELD();
                    self->state = EAT_COMMENT;
                } else {
                    // normal character - save in field
                    PUSH_CHAR(c);
                }
                break;

            case IN_QUOTED_FIELD:
                // in quoted field
                if (IS_ESCAPE_CHAR(c)) {
                    // possible escape character
                    self->state = ESCAPE_IN_QUOTED_FIELD;
                } else if (IS_QUOTE(c)) {
                    if (self->doublequote) {
                        // double quote - " represented by ""
                        self->state = QUOTE_IN_QUOTED_FIELD;
                    } else {
                        // end of quote part of field
                        self->state = IN_FIELD;
                    }
                } else {
                    // normal character - save in field
                    PUSH_CHAR(c);
                }
                break;

            case ESCAPE_IN_QUOTED_FIELD:
                PUSH_CHAR(c);
                self->state = IN_QUOTED_FIELD;
                break;

            case QUOTE_IN_QUOTED_FIELD:
                // double quote - seen a quote in an quoted field
                if (IS_QUOTE(c)) {
                    // save "" as "

                    PUSH_CHAR(c);
                    self->state = IN_QUOTED_FIELD;
                } else if (IS_DELIMITER(c)) {
                    // end of field - end of line not reached yet
                    END_FIELD();

                    if (self->delim_whitespace) {
                        self->state = EAT_WHITESPACE;
                    } else {
                        self->state = START_FIELD;
                    }
                } else if (IS_TERMINATOR(c)) {
                    END_FIELD();
                    END_LINE();
                } else if (IS_CARRIAGE(c)) {
                    END_FIELD();
                    self->state = EAT_CRNL;
                } else if (!self->strict) {
                    PUSH_CHAR(c);
                    self->state = IN_FIELD;
                } else {
                    int64_t bufsize = 100;
                    self->error_msg = (char *)malloc(bufsize);
                    snprintf(self->error_msg, bufsize,
                            "delimiter expected after quote in quote");
                    goto parsingerror;
                }
                break;

            case EAT_COMMENT:
                if (IS_TERMINATOR(c)) {
                    END_LINE();
                } else if (IS_CARRIAGE(c)) {
                    self->state = EAT_CRNL;
                }
                break;

            // only occurs with non-custom line terminator,
            // which is why we directly check for '\n'
            case EAT_CRNL:
                if (c == '\n') {
                    END_LINE();
                } else if (IS_DELIMITER(c)) {
                    if (self->delim_whitespace) {
                        END_LINE_STATE(EAT_WHITESPACE);
                    } else {
                        // Handle \r-delimited files
                        END_LINE_AND_FIELD_STATE(START_FIELD);
                    }
                } else {
                    if (self->delim_whitespace) {
                        /* XXX
                        * first character of a new record--need to back up and
                        * reread
                        * to handle properly...
                        */
                        i--;
                        buf--;  // back up one character (HACK!)
                        END_LINE_STATE(START_RECORD);
                    } else {
                        // \r line terminator
                        // UGH. we don't actually want
                        // to consume the token. fix this later
                        self->stream_len = slen;
                        if (end_line(self) < 0) {
                            goto parsingerror;
                        }

                        stream = self->stream + self->stream_len;
                        slen = self->stream_len;
                        self->state = START_RECORD;

                        --i;
                        buf--;  // let's try this character again (HACK!)
                        if (line_limit > 0 &&
                            self->lines == start_lines + line_limit) {
                            goto linelimit;
                        }
                    }
                }
                break;

            // only occurs with non-custom line terminator,
            // which is why we directly check for '\n'
            case EAT_CRNL_NOP:  // inside an ignored comment line
                self->state = START_RECORD;
                // \r line terminator -- parse this character again
                if (c != '\n' && !IS_DELIMITER(c)) {
                    --i;
                    --buf;
                }
                break;
            default:
                break;
        }
    }

    _TOKEN_CLEANUP();

    TRACE(("Finished tokenizing input\n"))

    return 0;

parsingerror:
    i++;
    _TOKEN_CLEANUP();

    return -1;

linelimit:
    i++;
    _TOKEN_CLEANUP();

    return 0;
}

static int parser_handle_eof(parser_t *self) {
    int64_t bufsize = 100;

    TRACE(
        ("handling eof, datalen: %d, pstate: %d\n", self->datalen, self->state))

    if (self->datalen != 0) return -1;

    switch (self->state) {
        case START_RECORD:
        case WHITESPACE_LINE:
        case EAT_CRNL_NOP:
        case EAT_LINE_COMMENT:
            return 0;

        case ESCAPE_IN_QUOTED_FIELD:
        case IN_QUOTED_FIELD:
            self->error_msg = (char *)malloc(bufsize);
            snprintf(self->error_msg, bufsize,
                    "EOF inside string starting at row %lld",
                    (long long)self->file_lines);
            return -1;

        case ESCAPED_CHAR:
            self->error_msg = (char *)malloc(bufsize);
            snprintf(self->error_msg, bufsize,
                     "EOF following escape character");
            return -1;

        case IN_FIELD:
        case START_FIELD:
        case QUOTE_IN_QUOTED_FIELD:
            if (end_field(self) < 0) return -1;
            break;

        default:
            break;
    }

    if (end_line(self) < 0)
        return -1;
    else
        return 0;
}

int parser_consume_rows(parser_t *self, size_t nrows) {
    int64_t i, offset, word_deletions, char_count;

    if (nrows > self->lines) {
        nrows = self->lines;
    }

    /* do nothing */
    if (nrows == 0) return 0;

    /* cannot guarantee that nrows + 1 has been observed */
    word_deletions = self->line_start[nrows - 1] + self->line_fields[nrows - 1];
    char_count = (self->word_starts[word_deletions - 1] +
                  strlen(self->words[word_deletions - 1]) + 1);

    TRACE(("parser_consume_rows: Deleting %d words, %d chars\n", word_deletions,
           char_count));

    /* move stream, only if something to move */
    if (char_count < self->stream_len) {
        memmove((void *)self->stream, (void *)(self->stream + char_count),
                self->stream_len - char_count);
    }
    /* buffer counts */
    self->stream_len -= char_count;

    /* move token metadata */
    for (i = 0; i < self->words_len - word_deletions; ++i) {
        offset = i + word_deletions;

        self->words[i] = self->words[offset] - char_count;
        self->word_starts[i] = self->word_starts[offset] - char_count;
    }
    self->words_len -= word_deletions;

    /* move current word pointer to stream */
    self->pword_start -= char_count;
    self->word_start -= char_count;

    /* move line metadata */
    for (i = 0; i < self->lines - nrows + 1; ++i) {
        offset = i + nrows;
        self->line_start[i] = self->line_start[offset] - word_deletions;
        self->line_fields[i] = self->line_fields[offset];
    }
    self->lines -= nrows;

    return 0;
}

static size_t _next_pow2(size_t sz) {
    size_t result = 1;
    while (result < sz) result *= 2;
    return result;
}

int parser_trim_buffers(parser_t *self) {
    /*
      Free memory
     */
    size_t new_cap;
    void *newptr;

    int64_t i;

    /**
     * Before we free up space and trim, we should
     * save how many words we saw when parsing, if
     * it exceeds the maximum number we saw before.
     *
     * This is important for when we read in chunks,
     * so that we can inform subsequent chunk parsing
     * as to how many words we could possibly see.
     */
    if (self->words_cap > self->max_words_cap) {
        self->max_words_cap = self->words_cap;
    }

    /* trim words, word_starts */
    new_cap = _next_pow2(self->words_len) + 1;
    if (new_cap < self->words_cap) {
        TRACE(("parser_trim_buffers: new_cap < self->words_cap\n"));
        newptr = safe_realloc((void *)self->words, new_cap * sizeof(char *));
        if (newptr == NULL) {
            return PARSER_OUT_OF_MEMORY;
        } else {
            self->words = (char **)newptr;
        }
        newptr = safe_realloc((void *)self->word_starts,
                              new_cap * sizeof(int64_t));
        if (newptr == NULL) {
            return PARSER_OUT_OF_MEMORY;
        } else {
            self->word_starts = (int64_t *)newptr;
            self->words_cap = new_cap;
        }
    }

    /* trim stream */
    new_cap = _next_pow2(self->stream_len) + 1;
    TRACE(
        ("parser_trim_buffers: new_cap = %zu, stream_cap = %zu, lines_cap = "
         "%zu\n",
         new_cap, self->stream_cap, self->lines_cap));
    if (new_cap < self->stream_cap) {
        TRACE(
            ("parser_trim_buffers: new_cap < self->stream_cap, calling "
             "safe_realloc\n"));
        newptr = safe_realloc((void *)self->stream, new_cap);
        if (newptr == NULL) {
            return PARSER_OUT_OF_MEMORY;
        } else {
            // Update the pointers in the self->words array (char **) if
            // `safe_realloc`
            //  moved the `self->stream` buffer. This block mirrors a similar
            //  block in
            //  `make_stream_space`.
            if (self->stream != newptr) {
                self->pword_start = (char *)newptr + self->word_start;

                for (i = 0; i < self->words_len; ++i) {
                    self->words[i] = (char *)newptr + self->word_starts[i];
                }
            }

            self->stream = newptr;
            self->stream_cap = new_cap;
        }
    }

    /* trim line_start, line_fields */
    new_cap = _next_pow2(self->lines) + 1;
    if (new_cap < self->lines_cap) {
        TRACE(("parser_trim_buffers: new_cap < self->lines_cap\n"));
        newptr = safe_realloc((void *)self->line_start,
                              new_cap * sizeof(int64_t));
        if (newptr == NULL) {
            return PARSER_OUT_OF_MEMORY;
        } else {
            self->line_start = (int64_t *)newptr;
        }
        newptr = safe_realloc((void *)self->line_fields,
                              new_cap * sizeof(int64_t));
        if (newptr == NULL) {
            return PARSER_OUT_OF_MEMORY;
        } else {
            self->line_fields = (int64_t *)newptr;
            self->lines_cap = new_cap;
        }
    }

    return 0;
}

/*
  nrows : number of rows to tokenize (or until reach EOF)
  all : tokenize all the data vs. certain number of rows
 */

int _tokenize_helper(parser_t *self, size_t nrows, int all) {
    int status = 0;
    int64_t start_lines = self->lines;

    if (self->state == FINISHED) {
        return 0;
    }

    TRACE((
        "_tokenize_helper: Asked to tokenize %d rows, datapos=%d, datalen=%d\n",
        nrows, self->datapos, self->datalen));

    while (1) {
        if (!all && self->lines - start_lines >= nrows) break;

        if (self->datapos == self->datalen) {
            status = parser_buffer_bytes(self, self->chunksize);

            if (status == REACHED_EOF) {
                // close out last line
                status = parser_handle_eof(self);
                self->state = FINISHED;
                break;
            } else if (status != 0) {
                return status;
            }
        }

        TRACE(
            ("_tokenize_helper: Trying to process %d bytes, datalen=%d, "
             "datapos= %d\n",
             self->datalen - self->datapos, self->datalen, self->datapos));

        status = tokenize_bytes(self, nrows, start_lines);

        if (status < 0) {
            // XXX
            TRACE(
                ("_tokenize_helper: Status %d returned from tokenize_bytes, "
                 "breaking\n",
                 status));
            status = -1;
            break;
        }
    }
    TRACE(("leaving tokenize_helper\n"));
    return status;
}

int tokenize_nrows(parser_t *self, size_t nrows) {
    int status = _tokenize_helper(self, nrows, 0);
    return status;
}

int tokenize_all_rows(parser_t *self) {
    int status = _tokenize_helper(self, -1, 1);
    return status;
}

PANDAS_INLINE void uppercase(char *p) {
    for (; *p; ++p) *p = toupper_ascii(*p);
}

int PANDAS_INLINE to_longlong(char *item, long long *p_value) {
    char *p_end;

    // Try integer conversion.  We explicitly give the base to be 10. If
    // we used 0, strtoll() would convert '012' to 10, because the leading 0 in
    // '012' signals an octal number in C.  For a general purpose reader, that
    // would be a bug, not a feature.
    *p_value = strtoll(item, &p_end, 10);

    // Allow trailing spaces.
    while (isspace_ascii(*p_end)) ++p_end;

    return (errno == 0) && (!*p_end);
}

int to_boolean(const char *item, uint8_t *val) {
    char *tmp;
    int i, status = 0;
    int bufsize = sizeof(char) * (strlen(item) + 1);

    static const char *tstrs[1] = {"TRUE"};
    static const char *fstrs[1] = {"FALSE"};

    tmp = malloc(bufsize);
    strncpy(tmp, item, bufsize);
    uppercase(tmp);

    for (i = 0; i < 1; ++i) {
        if (strcmp(tmp, tstrs[i]) == 0) {
            *val = 1;
            goto done;
        }
    }

    for (i = 0; i < 1; ++i) {
        if (strcmp(tmp, fstrs[i]) == 0) {
            *val = 0;
            goto done;
        }
    }

    status = -1;

done:
    free(tmp);
    return status;
}

#ifdef TEST

int main(int argc, char *argv[]) {
    double x, y;
    long long xi;
    int status;
    char *s;

    s = "123,789";
    status = to_longlong_thousands(s, &xi, ',');
    printf("s = '%s'\n", s);
    printf("status = %d\n", status);
    printf("x = %d\n", (int)xi);

    return 0;
}
#endif

// ---------------------------------------------------------------------------
// Implementation of xstrtod

//
// strtod.c
//
// Convert string to double
//
// Copyright (C) 2002 Michael Ringgaard. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//
// 1. Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
// 2. Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
// 3. Neither the name of the project nor the names of its contributors
//    may be used to endorse or promote products derived from this software
//    without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND
// ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
// LIABLE
// FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
// OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
// HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
// LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
// OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
// SUCH DAMAGE.
//
// -----------------------------------------------------------------------
// Modifications by Warren Weckesser, March 2011:
// * Rename strtod() to xstrtod().
// * Added decimal and sci arguments.
// * Skip trailing spaces.
// * Commented out the other functions.
// Modifications by Richard T Guy, August 2013:
// * Add tsep argument for thousands separator
//

double xstrtod(const char *str, char **endptr, char decimal, char sci,
               char tsep, int skip_trailing) {
    double number;
    int exponent;
    int negative;
    char *p = (char *)str;
    double p10;
    int n;
    int num_digits;
    int num_decimals;

    errno = 0;

    // Skip leading whitespace.
    while (isspace_ascii(*p)) p++;

    // Handle optional sign.
    negative = 0;
    switch (*p) {
        case '-':
            negative = 1;  // Fall through to increment position.
        case '+':
            p++;
    }

    number = 0.;
    exponent = 0;
    num_digits = 0;
    num_decimals = 0;

    // Process string of digits.
    while (isdigit_ascii(*p)) {
        number = number * 10. + (*p - '0');
        p++;
        num_digits++;

        p += (tsep != '\0' && *p == tsep);
    }

    // Process decimal part.
    if (*p == decimal) {
        p++;

        while (isdigit_ascii(*p)) {
            number = number * 10. + (*p - '0');
            p++;
            num_digits++;
            num_decimals++;
        }

        exponent -= num_decimals;
    }

    if (num_digits == 0) {
        errno = ERANGE;
        return 0.0;
    }

    // Correct for sign.
    if (negative) number = -number;

    // Process an exponent string.
    if (toupper_ascii(*p) == toupper_ascii(sci)) {
        // Handle optional sign.
        negative = 0;
        switch (*++p) {
            case '-':
                negative = 1;  // Fall through to increment pos.
            case '+':
                p++;
        }

        // Process string of digits.
        num_digits = 0;
        n = 0;
        while (isdigit_ascii(*p)) {
            n = n * 10 + (*p - '0');
            num_digits++;
            p++;
        }

        if (negative)
            exponent -= n;
        else
            exponent += n;

        // If no digits, after the 'e'/'E', un-consume it
        if (num_digits == 0) p--;
    }

    if (exponent < DBL_MIN_EXP || exponent > DBL_MAX_EXP) {
        errno = ERANGE;
        return HUGE_VAL;
    }

    // Scale the result.
    p10 = 10.;
    n = exponent;
    if (n < 0) n = -n;
    while (n) {
        if (n & 1) {
            if (exponent < 0)
                number /= p10;
            else
                number *= p10;
        }
        n >>= 1;
        p10 *= p10;
    }

    if (number == HUGE_VAL) {
        errno = ERANGE;
    }

    if (skip_trailing) {
        // Skip trailing whitespace.
        while (isspace_ascii(*p)) p++;
    }

    if (endptr) *endptr = p;

    return number;
}

double precise_xstrtod(const char *str, char **endptr, char decimal, char sci,
                       char tsep, int skip_trailing) {
    double number;
    int exponent;
    int negative;
    char *p = (char *)str;
    int num_digits;
    int num_decimals;
    int max_digits = 17;
    int n;
    // Cache powers of 10 in memory.
    static double e[] = {
        1.,    1e1,   1e2,   1e3,   1e4,   1e5,   1e6,   1e7,   1e8,   1e9,
        1e10,  1e11,  1e12,  1e13,  1e14,  1e15,  1e16,  1e17,  1e18,  1e19,
        1e20,  1e21,  1e22,  1e23,  1e24,  1e25,  1e26,  1e27,  1e28,  1e29,
        1e30,  1e31,  1e32,  1e33,  1e34,  1e35,  1e36,  1e37,  1e38,  1e39,
        1e40,  1e41,  1e42,  1e43,  1e44,  1e45,  1e46,  1e47,  1e48,  1e49,
        1e50,  1e51,  1e52,  1e53,  1e54,  1e55,  1e56,  1e57,  1e58,  1e59,
        1e60,  1e61,  1e62,  1e63,  1e64,  1e65,  1e66,  1e67,  1e68,  1e69,
        1e70,  1e71,  1e72,  1e73,  1e74,  1e75,  1e76,  1e77,  1e78,  1e79,
        1e80,  1e81,  1e82,  1e83,  1e84,  1e85,  1e86,  1e87,  1e88,  1e89,
        1e90,  1e91,  1e92,  1e93,  1e94,  1e95,  1e96,  1e97,  1e98,  1e99,
        1e100, 1e101, 1e102, 1e103, 1e104, 1e105, 1e106, 1e107, 1e108, 1e109,
        1e110, 1e111, 1e112, 1e113, 1e114, 1e115, 1e116, 1e117, 1e118, 1e119,
        1e120, 1e121, 1e122, 1e123, 1e124, 1e125, 1e126, 1e127, 1e128, 1e129,
        1e130, 1e131, 1e132, 1e133, 1e134, 1e135, 1e136, 1e137, 1e138, 1e139,
        1e140, 1e141, 1e142, 1e143, 1e144, 1e145, 1e146, 1e147, 1e148, 1e149,
        1e150, 1e151, 1e152, 1e153, 1e154, 1e155, 1e156, 1e157, 1e158, 1e159,
        1e160, 1e161, 1e162, 1e163, 1e164, 1e165, 1e166, 1e167, 1e168, 1e169,
        1e170, 1e171, 1e172, 1e173, 1e174, 1e175, 1e176, 1e177, 1e178, 1e179,
        1e180, 1e181, 1e182, 1e183, 1e184, 1e185, 1e186, 1e187, 1e188, 1e189,
        1e190, 1e191, 1e192, 1e193, 1e194, 1e195, 1e196, 1e197, 1e198, 1e199,
        1e200, 1e201, 1e202, 1e203, 1e204, 1e205, 1e206, 1e207, 1e208, 1e209,
        1e210, 1e211, 1e212, 1e213, 1e214, 1e215, 1e216, 1e217, 1e218, 1e219,
        1e220, 1e221, 1e222, 1e223, 1e224, 1e225, 1e226, 1e227, 1e228, 1e229,
        1e230, 1e231, 1e232, 1e233, 1e234, 1e235, 1e236, 1e237, 1e238, 1e239,
        1e240, 1e241, 1e242, 1e243, 1e244, 1e245, 1e246, 1e247, 1e248, 1e249,
        1e250, 1e251, 1e252, 1e253, 1e254, 1e255, 1e256, 1e257, 1e258, 1e259,
        1e260, 1e261, 1e262, 1e263, 1e264, 1e265, 1e266, 1e267, 1e268, 1e269,
        1e270, 1e271, 1e272, 1e273, 1e274, 1e275, 1e276, 1e277, 1e278, 1e279,
        1e280, 1e281, 1e282, 1e283, 1e284, 1e285, 1e286, 1e287, 1e288, 1e289,
        1e290, 1e291, 1e292, 1e293, 1e294, 1e295, 1e296, 1e297, 1e298, 1e299,
        1e300, 1e301, 1e302, 1e303, 1e304, 1e305, 1e306, 1e307, 1e308};
    errno = 0;

    // Skip leading whitespace.
    while (isspace_ascii(*p)) p++;

    // Handle optional sign.
    negative = 0;
    switch (*p) {
        case '-':
            negative = 1;  // Fall through to increment position.
        case '+':
            p++;
    }

    number = 0.;
    exponent = 0;
    num_digits = 0;
    num_decimals = 0;

    // Process string of digits.
    while (isdigit_ascii(*p)) {
        if (num_digits < max_digits) {
            number = number * 10. + (*p - '0');
            num_digits++;
        } else {
            ++exponent;
        }

        p++;
        p += (tsep != '\0' && *p == tsep);
    }

    // Process decimal part
    if (*p == decimal) {
        p++;

        while (num_digits < max_digits && isdigit_ascii(*p)) {
            number = number * 10. + (*p - '0');
            p++;
            num_digits++;
            num_decimals++;
        }

        if (num_digits >= max_digits)  // Consume extra decimal digits.
            while (isdigit_ascii(*p)) ++p;

        exponent -= num_decimals;
    }

    if (num_digits == 0) {
        errno = ERANGE;
        return 0.0;
    }

    // Correct for sign.
    if (negative) number = -number;

    // Process an exponent string.
    if (toupper_ascii(*p) == toupper_ascii(sci)) {
        // Handle optional sign
        negative = 0;
        switch (*++p) {
            case '-':
                negative = 1;  // Fall through to increment pos.
            case '+':
                p++;
        }

        // Process string of digits.
        num_digits = 0;
        n = 0;
        while (isdigit_ascii(*p)) {
            n = n * 10 + (*p - '0');
            num_digits++;
            p++;
        }

        if (negative)
            exponent -= n;
        else
            exponent += n;

        // If no digits after the 'e'/'E', un-consume it.
        if (num_digits == 0) p--;
    }

    if (exponent > 308) {
        errno = ERANGE;
        return HUGE_VAL;
    } else if (exponent > 0) {
        number *= e[exponent];
    } else if (exponent < -308) {  // Subnormal
        if (exponent < -616)       // Prevent invalid array access.
            number = 0.;
        number /= e[-308 - exponent];
        number /= e[308];
    } else {
        number /= e[-exponent];
    }

    if (number == HUGE_VAL || number == -HUGE_VAL) errno = ERANGE;

    if (skip_trailing) {
        // Skip trailing whitespace.
        while (isspace_ascii(*p)) p++;
    }

    if (endptr) *endptr = p;
    return number;
}

double round_trip(const char *p, char **q, char decimal, char sci, char tsep,
                  int skip_trailing) {
    double r = PyOS_string_to_double(p, q, 0);
    PyErr_Clear();
    return r;
}

// End of xstrtod code
// ---------------------------------------------------------------------------

void uint_state_init(uint_state *self) {
    self->seen_sint = 0;
    self->seen_uint = 0;
    self->seen_null = 0;
}

int uint64_conflict(uint_state *self) {
    return self->seen_uint && (self->seen_sint || self->seen_null);
}

int64_t str_to_int64(const char *p_item, int64_t int_min, int64_t int_max,
                     int *error, char tsep) {
    const char *p = (const char *)p_item;
    int isneg = 0;
    int64_t number = 0;
    int d;

    // Skip leading spaces.
    while (isspace_ascii(*p)) {
        ++p;
    }

    // Handle sign.
    if (*p == '-') {
        isneg = 1;
        ++p;
    } else if (*p == '+') {
        p++;
    }

    // Check that there is a first digit.
    if (!isdigit_ascii(*p)) {
        // Error...
        *error = ERROR_NO_DIGITS;
        return 0;
    }

    if (isneg) {
        // If number is greater than pre_min, at least one more digit
        // can be processed without overflowing.
        int dig_pre_min = -(int_min % 10);
        int64_t pre_min = int_min / 10;

        // Process the digits.
        d = *p;
        if (tsep != '\0') {
            while (1) {
                if (d == tsep) {
                    d = *++p;
                    continue;
                } else if (!isdigit_ascii(d)) {
                    break;
                }
                if ((number > pre_min) ||
                    ((number == pre_min) && (d - '0' <= dig_pre_min))) {
                    number = number * 10 - (d - '0');
                    d = *++p;
                } else {
                    *error = ERROR_OVERFLOW;
                    return 0;
                }
            }
        } else {
            while (isdigit_ascii(d)) {
                if ((number > pre_min) ||
                    ((number == pre_min) && (d - '0' <= dig_pre_min))) {
                    number = number * 10 - (d - '0');
                    d = *++p;
                } else {
                    *error = ERROR_OVERFLOW;
                    return 0;
                }
            }
        }
    } else {
        // If number is less than pre_max, at least one more digit
        // can be processed without overflowing.
        int64_t pre_max = int_max / 10;
        int dig_pre_max = int_max % 10;

        // Process the digits.
        d = *p;
        if (tsep != '\0') {
            while (1) {
                if (d == tsep) {
                    d = *++p;
                    continue;
                } else if (!isdigit_ascii(d)) {
                    break;
                }
                if ((number < pre_max) ||
                    ((number == pre_max) && (d - '0' <= dig_pre_max))) {
                    number = number * 10 + (d - '0');
                    d = *++p;

                } else {
                    *error = ERROR_OVERFLOW;
                    return 0;
                }
            }
        } else {
            while (isdigit_ascii(d)) {
                if ((number < pre_max) ||
                    ((number == pre_max) && (d - '0' <= dig_pre_max))) {
                    number = number * 10 + (d - '0');
                    d = *++p;

                } else {
                    *error = ERROR_OVERFLOW;
                    return 0;
                }
            }
        }
    }

    // Skip trailing spaces.
    while (isspace_ascii(*p)) {
        ++p;
    }

    // Did we use up all the characters?
    if (*p) {
        *error = ERROR_INVALID_CHARS;
        return 0;
    }

    *error = 0;
    return number;
}

uint64_t str_to_uint64(uint_state *state, const char *p_item, int64_t int_max,
                       uint64_t uint_max, int *error, char tsep) {
    const char *p = (const char *)p_item;
    uint64_t pre_max = uint_max / 10;
    int dig_pre_max = uint_max % 10;
    uint64_t number = 0;
    int d;

    // Skip leading spaces.
    while (isspace_ascii(*p)) {
        ++p;
    }

    // Handle sign.
    if (*p == '-') {
        state->seen_sint = 1;
        *error = 0;
        return 0;
    } else if (*p == '+') {
        p++;
    }

    // Check that there is a first digit.
    if (!isdigit_ascii(*p)) {
        // Error...
        *error = ERROR_NO_DIGITS;
        return 0;
    }

    // If number is less than pre_max, at least one more digit
    // can be processed without overflowing.
    //
    // Process the digits.
    d = *p;
    if (tsep != '\0') {
        while (1) {
            if (d == tsep) {
                d = *++p;
                continue;
            } else if (!isdigit_ascii(d)) {
                break;
            }
            if ((number < pre_max) ||
                ((number == pre_max) && (d - '0' <= dig_pre_max))) {
                number = number * 10 + (d - '0');
                d = *++p;

            } else {
                *error = ERROR_OVERFLOW;
                return 0;
            }
        }
    } else {
        while (isdigit_ascii(d)) {
            if ((number < pre_max) ||
                ((number == pre_max) && (d - '0' <= dig_pre_max))) {
                number = number * 10 + (d - '0');
                d = *++p;

            } else {
                *error = ERROR_OVERFLOW;
                return 0;
            }
        }
    }

    // Skip trailing spaces.
    while (isspace_ascii(*p)) {
        ++p;
    }

    // Did we use up all the characters?
    if (*p) {
        *error = ERROR_INVALID_CHARS;
        return 0;
    }

    if (number > (uint64_t)int_max) {
        state->seen_uint = 1;
    }

    *error = 0;
    return number;
}
