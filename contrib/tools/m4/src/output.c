/* GNU m4 -- A simple macro processor

   Copyright (C) 1989-1994, 2004-2013 Free Software Foundation, Inc.

   This file is part of GNU M4.

   GNU M4 is free software: you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.

   GNU M4 is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.

   You should have received a copy of the GNU General Public License
   along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#include "m4.h"

#include <limits.h>
#include <sys/stat.h>

#include "gl_avltree_oset.h"
#include "gl_xoset.h"

/* Size of initial in-memory buffer size for diversions.  Small diversions
   would usually fit in.  */
#define INITIAL_BUFFER_SIZE 512

/* Maximum value for the total of all in-memory buffer sizes for
   diversions.  */
#define MAXIMUM_TOTAL_SIZE (512 * 1024)

/* Size of buffer size to use while copying files.  */
#define COPY_BUFFER_SIZE (32 * 512)

/* Output functions.  Most of the complexity is for handling cpp like
   sync lines.

   This code is fairly entangled with the code in input.c, and maybe it
   belongs there?  */

typedef struct temp_dir m4_temp_dir;

/* When part of diversion_table, each struct m4_diversion either
   represents an open file (zero size, non-NULL u.file), an in-memory
   buffer (non-zero size, non-NULL u.buffer), or an unused placeholder
   diversion (zero size, u is NULL, non-zero used indicates that a
   file has been created).  When not part of diversion_table, u.next
   is a pointer to the free_list chain.  */

typedef struct m4_diversion m4_diversion;

struct m4_diversion
  {
    union
      {
        FILE *file;             /* Diversion file on disk.  */
        char *buffer;           /* Malloc'd diversion buffer.  */
        m4_diversion *next;     /* Free-list pointer */
      } u;
    int divnum;                 /* Which diversion this represents.  */
    int size;                   /* Usable size before reallocation.  */
    int used;                   /* Used buffer length, or tmp file exists.  */
  };

/* Table of diversions 1 through INT_MAX.  */
static gl_oset_t diversion_table;

/* Diversion 0 (not part of diversion_table).  */
static m4_diversion div0;

/* Linked list of reclaimed diversion storage.  */
static m4_diversion *free_list;

/* Obstack from which diversion storage is allocated.  */
static struct obstack diversion_storage;

/* Total size of all in-memory buffer sizes.  */
static int total_buffer_size;

/* The number of the currently active diversion.  This variable is
   maintained for the `divnum' builtin function.  */
int current_diversion;

/* Current output diversion, NULL if output is being currently
   discarded.  output_diversion->u is guaranteed non-NULL except when
   the diversion has never been used; use size to determine if it is a
   malloc'd buffer or a FILE.  output_diversion->used is 0 if u.file
   is stdout, and non-zero if this is a malloc'd buffer or a temporary
   diversion file.  */
static m4_diversion *output_diversion;

/* Cache of output_diversion->u.file, only valid when
   output_diversion->size is 0.  */
static FILE *output_file;

/* Cache of output_diversion->u.buffer + output_diversion->used, only
   valid when output_diversion->size is non-zero.  */
static char *output_cursor;

/* Cache of output_diversion->size - output_diversion->used, only
   valid when output_diversion->size is non-zero.  */
static int output_unused;

/* Number of input line we are generating output for.  */
int output_current_line;

/* Temporary directory holding all spilled diversion files.  */
static m4_temp_dir *output_temp_dir;

/* Cache of most recently used spilled diversion files.  */
static FILE *tmp_file1;
static FILE *tmp_file2;

/* Diversions that own tmp_file, or 0.  */
static int tmp_file1_owner;
static int tmp_file2_owner;

/* True if tmp_file2 is more recently used.  */
static bool tmp_file2_recent;


/* Internal routines.  */

/* Callback for comparing list elements ELT1 and ELT2 for order in
   diversion_table.  */
static int
cmp_diversion_CB (const void *elt1, const void *elt2)
{
  const m4_diversion *d1 = (const m4_diversion *) elt1;
  const m4_diversion *d2 = (const m4_diversion *) elt2;
  /* No need to worry about overflow, since we don't create diversions
     with negative divnum.  */
  return d1->divnum - d2->divnum;
}

/* Callback for comparing list element ELT against THRESHOLD.  */
static bool
threshold_diversion_CB (const void *elt, const void *threshold)
{
  const m4_diversion *diversion = (const m4_diversion *) elt;
  /* No need to worry about overflow, since we don't create diversions
     with negative divnum.  */
  return diversion->divnum >= *(const int *) threshold;
}

/* Clean up any temporary directory.  Designed for use as an atexit
   handler, where it is not safe to call exit() recursively; so this
   calls _exit if a problem is encountered.  */
static void
cleanup_tmpfile (void)
{
  /* Close any open diversions.  */
  bool fail = false;

  if (diversion_table)
    {
      const void *elt;
      gl_oset_iterator_t iter = gl_oset_iterator (diversion_table);
      while (gl_oset_iterator_next (&iter, &elt))
        {
          m4_diversion *diversion = (m4_diversion *) elt;
          if (!diversion->size && diversion->u.file
              && close_stream_temp (diversion->u.file) != 0)
            {
              M4ERROR ((0, errno,
                        "cannot clean temporary file for diversion"));
              fail = true;
            }
        }
      gl_oset_iterator_free (&iter);
    }

  /* Clean up the temporary directory.  */
  if (cleanup_temp_dir (output_temp_dir) != 0)
    fail = true;
  if (fail)
    _exit (exit_failure);
}

/* Convert DIVNUM into a temporary file name for use in m4_tmp*.  */
static const char *
m4_tmpname (int divnum)
{
  static char *buffer;
  static char *tail;
  if (buffer == NULL)
    {
      tail = xasprintf ("%s/m4-%d", output_temp_dir->dir_name, INT_MAX);
      buffer = (char *) obstack_copy0 (&diversion_storage, tail,
                                       strlen (tail));
      free (tail);
      tail = strrchr (buffer, '-') + 1;
    }
  assert (0 < divnum);
  sprintf (tail, "%d", divnum);
  return buffer;
}

/* Create a temporary file for diversion DIVNUM open for reading and
   writing in a secure temp directory.  The file will be automatically
   closed and deleted on a fatal signal.  The file can be closed and
   reopened with m4_tmpclose and m4_tmpopen, or moved with
   m4_tmprename; when finally done with the file, close it with
   m4_tmpremove.  Exits on failure, so the return value is always an
   open file.  */
static FILE *
m4_tmpfile (int divnum)
{
  const char *name;
  FILE *file;

  if (output_temp_dir == NULL)
    {
      output_temp_dir = create_temp_dir ("m4-", NULL, true);
      if (output_temp_dir == NULL)
        M4ERROR ((EXIT_FAILURE, errno,
                  "cannot create temporary file for diversion"));
      atexit (cleanup_tmpfile);
    }
  name = m4_tmpname (divnum);
  register_temp_file (output_temp_dir, name);
  file = fopen_temp (name, O_BINARY ? "wb+" : "w+");
  if (file == NULL)
    {
      unregister_temp_file (output_temp_dir, name);
      M4ERROR ((EXIT_FAILURE, errno,
                "cannot create temporary file for diversion"));
    }
  else if (set_cloexec_flag (fileno (file), true) != 0)
    M4ERROR ((warning_status, errno,
              "Warning: cannot protect diversion across forks"));
  return file;
}

/* Reopen a temporary file for diversion DIVNUM for reading and
   writing in a secure temp directory.  If REREAD, the file is
   positioned at offset 0, otherwise the file is positioned at the
   end.  Exits on failure, so the return value is always an open
   file.  */
static FILE *
m4_tmpopen (int divnum, bool reread)
{
  const char *name;
  FILE *file;

  if (tmp_file1_owner == divnum)
    {
      if (reread && fseeko (tmp_file1, 0, SEEK_SET) != 0)
        m4_error (EXIT_FAILURE, errno,
                  _("cannot seek within diversion"));
      tmp_file2_recent = false;
      return tmp_file1;
    }
  else if (tmp_file2_owner == divnum)
    {
      if (reread && fseeko (tmp_file2, 0, SEEK_SET) != 0)
        m4_error (EXIT_FAILURE, errno,
                  _("cannot seek within diversion"));
      tmp_file2_recent = true;
      return tmp_file2;
    }
  name = m4_tmpname (divnum);
  /* We need update mode, to avoid truncation.  */
  file = fopen_temp (name, O_BINARY ? "rb+" : "r+");
  if (file == NULL)
    M4ERROR ((EXIT_FAILURE, errno,
              "cannot create temporary file for diversion"));
  else if (set_cloexec_flag (fileno (file), true) != 0)
    m4_error (0, errno, _("cannot protect diversion across forks"));
  /* Update mode starts at the beginning of the stream, but sometimes
     we want the end.  */
  else if (!reread && fseeko (file, 0, SEEK_END) != 0)
    m4_error (EXIT_FAILURE, errno,
              _("cannot seek within diversion"));
  return file;
}

/* Close, but don't delete, a temporary FILE for diversion DIVNUM.  To
   reduce the I/O overhead of repeatedly opening and closing the same
   file, this implementation caches the most recent spilled diversion.
   On the other hand, keeping every spilled diversion open would run
   into EMFILE limits.  */
static int
m4_tmpclose (FILE *file, int divnum)
{
  int result = 0;
  if (divnum != tmp_file1_owner && divnum != tmp_file2_owner)
    {
      if (tmp_file2_recent)
        {
          if (tmp_file1_owner)
            result = close_stream_temp (tmp_file1);
          tmp_file1 = file;
          tmp_file1_owner = divnum;
        }
      else
        {
          if (tmp_file2_owner)
            result = close_stream_temp (tmp_file2);
          tmp_file2 = file;
          tmp_file2_owner = divnum;
        }
    }
  return result;
}

/* Delete a closed temporary FILE for diversion DIVNUM.  */
static int
m4_tmpremove (int divnum)
{
  if (divnum == tmp_file1_owner)
    {
      int result = close_stream_temp (tmp_file1);
      if (result)
        return result;
      tmp_file1_owner = 0;
    }
  else if (divnum == tmp_file2_owner)
    {
      int result = close_stream_temp (tmp_file2);
      if (result)
        return result;
      tmp_file2_owner = 0;
    }
  return cleanup_temp_file (output_temp_dir, m4_tmpname (divnum));
}

/* Transfer the temporary file for diversion OLDNUM to the previously
   unused diversion NEWNUM.  Return an open stream visiting the new
   temporary file, positioned at the end, or exit on failure.  */
static FILE*
m4_tmprename (int oldnum, int newnum)
{
  /* m4_tmpname reuses its return buffer.  */
  char *oldname = xstrdup (m4_tmpname (oldnum));
  const char *newname = m4_tmpname (newnum);
  register_temp_file (output_temp_dir, newname);
  if (oldnum == tmp_file1_owner)
    {
      /* Be careful of mingw, which can't rename an open file.  */
      if (RENAME_OPEN_FILE_WORKS)
        tmp_file1_owner = newnum;
      else
        {
          if (close_stream_temp (tmp_file1))
            m4_error (EXIT_FAILURE, errno,
                      _("cannot close temporary file for diversion"));
          tmp_file1_owner = 0;
        }
    }
  else if (oldnum == tmp_file2_owner)
    {
      /* Be careful of mingw, which can't rename an open file.  */
      if (RENAME_OPEN_FILE_WORKS)
        tmp_file2_owner = newnum;
      else
        {
          if (close_stream_temp (tmp_file2))
            m4_error (EXIT_FAILURE, errno,
                      _("cannot close temporary file for diversion"));
          tmp_file2_owner = 0;
        }
    }
  /* Either it is safe to rename an open file, or no one should have
     oldname open at this point.  */
  if (rename (oldname, newname))
    m4_error (EXIT_FAILURE, errno,
              _("cannot create temporary file for diversion"));
  unregister_temp_file (output_temp_dir, oldname);
  free (oldname);
  return m4_tmpopen (newnum, false);
}


/*------------------------.
| Output initialization.  |
`------------------------*/

void
output_init (void)
{
  diversion_table = gl_oset_create_empty (GL_AVLTREE_OSET, cmp_diversion_CB,
                                          NULL);
  div0.u.file = stdout;
  output_diversion = &div0;
  output_file = stdout;
  obstack_init (&diversion_storage);
}

void
output_exit (void)
{
  /* Order is important, since we may have registered cleanup_tmpfile
     as an atexit handler, and it must not traverse stale memory.  */
  gl_oset_t table = diversion_table;
  if (tmp_file1_owner)
    m4_tmpremove (tmp_file1_owner);
  if (tmp_file2_owner)
    m4_tmpremove (tmp_file2_owner);
  diversion_table = NULL;
  gl_oset_free (table);
  obstack_free (&diversion_storage, NULL);
}

/*----------------------------------------------------------------.
| Reorganize in-memory diversion buffers so the current diversion |
| can accomodate LENGTH more characters without further           |
| reorganization.  The current diversion buffer is made bigger if |
| possible.  But to make room for a bigger buffer, one of the     |
| in-memory diversion buffers might have to be flushed to a newly |
| created temporary file.  This flushed buffer might well be the  |
| current one.                                                    |
`----------------------------------------------------------------*/

static void
make_room_for (int length)
{
  int wanted_size;
  m4_diversion *selected_diversion = NULL;

  /* Compute needed size for in-memory buffer.  Diversions in-memory
     buffers start at 0 bytes, then 512, then keep doubling until it is
     decided to flush them to disk.  */

  output_diversion->used = output_diversion->size - output_unused;

  for (wanted_size = output_diversion->size;
       wanted_size < output_diversion->used + length;
       wanted_size = wanted_size == 0 ? INITIAL_BUFFER_SIZE : wanted_size * 2)
    ;

  /* Check if we are exceeding the maximum amount of buffer memory.  */

  if (total_buffer_size - output_diversion->size + wanted_size
      > MAXIMUM_TOTAL_SIZE)
    {
      int selected_used;
      char *selected_buffer;
      m4_diversion *diversion;
      int count;
      gl_oset_iterator_t iter;
      const void *elt;

      /* Find out the buffer having most data, in view of flushing it to
         disk.  Fake the current buffer as having already received the
         projected data, while making the selection.  So, if it is
         selected indeed, we will flush it smaller, before it grows.  */

      selected_diversion = output_diversion;
      selected_used = output_diversion->used + length;

      iter = gl_oset_iterator (diversion_table);
      while (gl_oset_iterator_next (&iter, &elt))
        {
          diversion = (m4_diversion *) elt;
          if (diversion->used > selected_used)
            {
              selected_diversion = diversion;
              selected_used = diversion->used;
            }
        }
      gl_oset_iterator_free (&iter);

      /* Create a temporary file, write the in-memory buffer of the
         diversion to this file, then release the buffer.  Zero the
         diversion before doing anything that can exit () (including
         m4_tmpfile), so that the atexit handler doesn't try to close
         a garbage pointer as a file.  */

      selected_buffer = selected_diversion->u.buffer;
      total_buffer_size -= selected_diversion->size;
      selected_diversion->size = 0;
      selected_diversion->u.file = NULL;
      selected_diversion->u.file = m4_tmpfile (selected_diversion->divnum);

      if (selected_diversion->used > 0)
        {
          count = fwrite (selected_buffer, (size_t) selected_diversion->used,
                          1, selected_diversion->u.file);
          if (count != 1)
            M4ERROR ((EXIT_FAILURE, errno,
                      "ERROR: cannot flush diversion to temporary file"));
        }

      /* Reclaim the buffer space for other diversions.  */

      free (selected_buffer);
      selected_diversion->used = 1;
    }

  /* Reload output_file, just in case the flushed diversion was current.  */

  if (output_diversion == selected_diversion)
    {
      /* The flushed diversion was current indeed.  */

      output_file = output_diversion->u.file;
      output_cursor = NULL;
      output_unused = 0;
    }
  else
    {
      /* Close any selected file since it is not the current diversion.  */
      if (selected_diversion)
        {
          FILE *file = selected_diversion->u.file;
          selected_diversion->u.file = NULL;
          if (m4_tmpclose (file, selected_diversion->divnum) != 0)
            m4_error (0, errno,
                      _("cannot close temporary file for diversion"));
        }

      /* The current buffer may be safely reallocated.  */
      {
        char *buffer = output_diversion->u.buffer;
        output_diversion->u.buffer = xcharalloc ((size_t) wanted_size);
        memcpy (output_diversion->u.buffer, buffer, output_diversion->used);
        free (buffer);
      }

      total_buffer_size += wanted_size - output_diversion->size;
      output_diversion->size = wanted_size;

      output_cursor = output_diversion->u.buffer + output_diversion->used;
      output_unused = wanted_size - output_diversion->used;
    }
}

/*--------------------------------------------------------------.
| Output one character CHAR, when it is known that it goes to a |
| diversion file or an in-memory diversion buffer.              |
`--------------------------------------------------------------*/

#define OUTPUT_CHARACTER(Char) \
  if (output_file)                                                      \
    putc ((Char), output_file);                                         \
  else if (output_unused == 0)                                          \
    output_character_helper ((Char));                                   \
  else                                                                  \
    (output_unused--, *output_cursor++ = (Char))

static void
output_character_helper (int character)
{
  make_room_for (1);

  if (output_file)
    putc (character, output_file);
  else
    {
      *output_cursor++ = character;
      output_unused--;
    }
}

/*-------------------------------------------------------------------.
| Output one TEXT having LENGTH characters, when it is known that it |
| goes to a diversion file or an in-memory diversion buffer.         |
`-------------------------------------------------------------------*/

void
output_text (const char *text, int length)
{
  int count;

  if (!output_diversion || !length)
    return;

  if (!output_file && length > output_unused)
    make_room_for (length);

  if (output_file)
    {
      count = fwrite (text, length, 1, output_file);
      if (count != 1)
        M4ERROR ((EXIT_FAILURE, errno, "ERROR: copying inserted file"));
    }
  else
    {
      memcpy (output_cursor, text, (size_t) length);
      output_cursor += length;
      output_unused -= length;
    }
}

/*--------------------------------------------------------------------.
| Add some text into an obstack OBS, taken from TEXT, having LENGTH   |
| characters.  If OBS is NULL, output the text to an external file    |
| or an in-memory diversion buffer instead.  If OBS is NULL, and      |
| there is no output file, the text is discarded.  LINE is the line   |
| where the token starts (not necessarily current_line, in the case   |
| of multiline tokens).                                               |
|                                                                     |
| If we are generating sync lines, the output has to be examined,     |
| because we need to know how much output each input line generates.  |
| In general, sync lines are output whenever a single input lines     |
| generates several output lines, or when several input lines do not  |
| generate any output.                                                |
`--------------------------------------------------------------------*/

void
shipout_text (struct obstack *obs, const char *text, int length, int line)
{
  static bool start_of_output_line = true;
  const char *cursor;

  /* If output goes to an obstack, merely add TEXT to it.  */

  if (obs != NULL)
    {
      obstack_grow (obs, text, length);
      return;
    }

  /* Do nothing if TEXT should be discarded.  */

  if (output_diversion == NULL)
    return;

  /* Output TEXT to a file, or in-memory diversion buffer.  */

  if (!sync_output)
    switch (length)
      {

        /* In-line short texts.  */

      case 8: OUTPUT_CHARACTER (*text); text++;
      case 7: OUTPUT_CHARACTER (*text); text++;
      case 6: OUTPUT_CHARACTER (*text); text++;
      case 5: OUTPUT_CHARACTER (*text); text++;
      case 4: OUTPUT_CHARACTER (*text); text++;
      case 3: OUTPUT_CHARACTER (*text); text++;
      case 2: OUTPUT_CHARACTER (*text); text++;
      case 1: OUTPUT_CHARACTER (*text);
      case 0:
        return;

        /* Optimize longer texts.  */

      default:
        output_text (text, length);
      }
  else
    {
      /* Check for syncline only at the start of a token.  Multiline
         tokens, and tokens that are out of sync but in the middle of
         the line, must wait until the next raw newline triggers a
         syncline.  */
      if (start_of_output_line)
        {
          start_of_output_line = false;
          output_current_line++;
#ifdef DEBUG_OUTPUT
          xfprintf (stderr, "DEBUG: line %d, cur %d, cur out %d\n",
                   line, current_line, output_current_line);
#endif

          /* Output a `#line NUM' synchronization directive if needed.
             If output_current_line was previously given a negative
             value (invalidated), output `#line NUM "FILE"' instead.  */

          if (output_current_line != line)
            {
              OUTPUT_CHARACTER ('#');
              OUTPUT_CHARACTER ('l');
              OUTPUT_CHARACTER ('i');
              OUTPUT_CHARACTER ('n');
              OUTPUT_CHARACTER ('e');
              OUTPUT_CHARACTER (' ');
              for (cursor = ntoa (line, 10); *cursor; cursor++)
                OUTPUT_CHARACTER (*cursor);
              if (output_current_line < 1 && current_file[0] != '\0')
                {
                  OUTPUT_CHARACTER (' ');
                  OUTPUT_CHARACTER ('"');
                  for (cursor = current_file; *cursor; cursor++)
                    OUTPUT_CHARACTER (*cursor);
                  OUTPUT_CHARACTER ('"');
                }
              OUTPUT_CHARACTER ('\n');
              output_current_line = line;
            }
        }

      /* Output the token, and track embedded newlines.  */
      for (; length-- > 0; text++)
        {
          if (start_of_output_line)
            {
              start_of_output_line = false;
              output_current_line++;
#ifdef DEBUG_OUTPUT
              xfprintf (stderr, "DEBUG: line %d, cur %d, cur out %d\n",
                       line, current_line, output_current_line);
#endif
            }
          OUTPUT_CHARACTER (*text);
          if (*text == '\n')
            start_of_output_line = true;
        }
    }
}

/* Functions for use by diversions.  */

/*------------------------------------------------------------------.
| Make a file for diversion DIVNUM, and install it in the diversion |
| table.  Grow the size of the diversion table as needed.           |
`------------------------------------------------------------------*/

/* The number of possible diversions is limited only by memory and
   available file descriptors (each overflowing diversion uses one).  */

void
make_diversion (int divnum)
{
  m4_diversion *diversion = NULL;

  if (current_diversion == divnum)
    return;

  if (output_diversion)
    {
      if (!output_diversion->size && !output_diversion->u.file)
        {
          assert (!output_diversion->used);
          if (!gl_oset_remove (diversion_table, output_diversion))
            assert (false);
          output_diversion->u.next = free_list;
          free_list = output_diversion;
        }
      else if (output_diversion->size)
        output_diversion->used = output_diversion->size - output_unused;
      else if (output_diversion->used)
        {
          FILE *file = output_diversion->u.file;
          output_diversion->u.file = NULL;
          if (m4_tmpclose (file, output_diversion->divnum) != 0)
            m4_error (0, errno,
                      _("cannot close temporary file for diversion"));
        }
      output_diversion = NULL;
      output_file = NULL;
      output_cursor = NULL;
      output_unused = 0;
    }

  current_diversion = divnum;

  if (divnum < 0)
    return;

  if (divnum == 0)
    diversion = &div0;
  else
    {
      const void *elt;
      if (gl_oset_search_atleast (diversion_table, threshold_diversion_CB,
                                  &divnum, &elt))
        {
          m4_diversion *temp = (m4_diversion *) elt;
          if (temp->divnum == divnum)
            diversion = temp;
        }
    }
  if (diversion == NULL)
    {
      /* First time visiting this diversion.  */
      if (free_list)
        {
          diversion = free_list;
          free_list = diversion->u.next;
        }
      else
        {
          diversion = (m4_diversion *) obstack_alloc (&diversion_storage,
                                                      sizeof *diversion);
          diversion->size = 0;
          diversion->used = 0;
        }
      diversion->u.file = NULL;
      diversion->divnum = divnum;
      gl_oset_add (diversion_table, diversion);
    }

  output_diversion = diversion;
  if (output_diversion->size)
    {
      output_cursor = output_diversion->u.buffer + output_diversion->used;
      output_unused = output_diversion->size - output_diversion->used;
    }
  else
    {
      if (!output_diversion->u.file && output_diversion->used)
        output_diversion->u.file = m4_tmpopen (output_diversion->divnum,
                                               false);
      output_file = output_diversion->u.file;
    }
  output_current_line = -1;
}

/*-------------------------------------------------------------------.
| Insert a FILE into the current output file, in the same manner     |
| diversions are handled.  This allows files to be included, without |
| having them rescanned by m4.                                       |
`-------------------------------------------------------------------*/

void
insert_file (FILE *file)
{
  static char buffer[COPY_BUFFER_SIZE];
  size_t length;

  /* Optimize out inserting into a sink.  */
  if (!output_diversion)
    return;

  /* Insert output by big chunks.  */
  while (1)
    {
      length = fread (buffer, 1, sizeof buffer, file);
      if (ferror (file))
        M4ERROR ((EXIT_FAILURE, errno, "error reading inserted file"));
      if (length == 0)
        break;
      output_text (buffer, length);
    }
}

/*-------------------------------------------------------------------.
| Insert DIVERSION (but not div0) into the current output file.  The |
| diversion is NOT placed on the expansion obstack, because it must  |
| not be rescanned.  When the file is closed, it is deleted by the   |
| system.                                                            |
`-------------------------------------------------------------------*/

static void
insert_diversion_helper (m4_diversion *diversion)
{
  /* Effectively undivert only if an output stream is active.  */
  if (output_diversion)
    {
      if (diversion->size)
        {
          if (!output_diversion->u.file)
            {
              /* Transferring diversion metadata is faster than
                 copying contents.  */
              assert (!output_diversion->used && output_diversion != &div0
                      && !output_file);
              output_diversion->u.buffer = diversion->u.buffer;
              output_diversion->size = diversion->size;
              output_cursor = diversion->u.buffer + diversion->used;
              output_unused = diversion->size - diversion->used;
              diversion->u.buffer = NULL;
            }
          else
            {
              /* Avoid double-charging the total in-memory size when
                 transferring from one in-memory diversion to
                 another.  */
              total_buffer_size -= diversion->size;
              output_text (diversion->u.buffer, diversion->used);
            }
        }
      else if (!output_diversion->u.file)
        {
          /* Transferring diversion metadata is faster than copying
             contents.  */
          assert (!output_diversion->used && output_diversion != &div0
                  && !output_file);
          output_diversion->u.file = m4_tmprename (diversion->divnum,
                                                   output_diversion->divnum);
          output_diversion->used = 1;
          output_file = output_diversion->u.file;
          diversion->u.file = NULL;
          diversion->size = 1;
        }
      else
        {
          if (!diversion->u.file)
            diversion->u.file = m4_tmpopen (diversion->divnum, true);
          insert_file (diversion->u.file);
        }

      output_current_line = -1;
    }

  /* Return all space used by the diversion.  */
  if (diversion->size)
    {
      if (!output_diversion)
        total_buffer_size -= diversion->size;
      free (diversion->u.buffer);
      diversion->size = 0;
    }
  else
    {
      if (diversion->u.file)
        {
          FILE *file = diversion->u.file;
          diversion->u.file = NULL;
          if (m4_tmpclose (file, diversion->divnum) != 0)
            m4_error (0, errno,
                      _("cannot clean temporary file for diversion"));
        }
      if (m4_tmpremove (diversion->divnum) != 0)
        M4ERROR ((0, errno, "cannot clean temporary file for diversion"));
    }
  diversion->used = 0;
  gl_oset_remove (diversion_table, diversion);
  diversion->u.next = free_list;
  free_list = diversion;
}

/*------------------------------------------------------------------.
| Insert diversion number DIVNUM into the current output file.  The |
| diversion is NOT placed on the expansion obstack, because it must |
| not be rescanned.  When the file is closed, it is deleted by the  |
| system.                                                           |
`------------------------------------------------------------------*/

void
insert_diversion (int divnum)
{
  const void *elt;

  /* Do not care about nonexistent diversions, and undiverting stdout
     or self is a no-op.  */
  if (divnum <= 0 || current_diversion == divnum)
    return;
  if (gl_oset_search_atleast (diversion_table, threshold_diversion_CB,
                              &divnum, &elt))
    {
      m4_diversion *diversion = (m4_diversion *) elt;
      if (diversion->divnum == divnum)
        insert_diversion_helper (diversion);
    }
}

/*----------------------------------------------------------------.
| Get back all diversions.  This is done just before exiting from |
| main, and from m4_undivert (), if called without arguments.     |
`----------------------------------------------------------------*/

void
undivert_all (void)
{
  const void *elt;
  gl_oset_iterator_t iter = gl_oset_iterator (diversion_table);
  while (gl_oset_iterator_next (&iter, &elt))
    {
      m4_diversion *diversion = (m4_diversion *) elt;
      if (diversion->divnum != current_diversion)
        insert_diversion_helper (diversion);
    }
  gl_oset_iterator_free (&iter);
}

/*-------------------------------------------------------------.
| Produce all diversion information in frozen format on FILE.  |
`-------------------------------------------------------------*/

void
freeze_diversions (FILE *file)
{
  int saved_number;
  int last_inserted;
  gl_oset_iterator_t iter;
  const void *elt;

  saved_number = current_diversion;
  last_inserted = 0;
  make_diversion (0);
  output_file = file; /* kludge in the frozen file */

  iter = gl_oset_iterator (diversion_table);
  while (gl_oset_iterator_next (&iter, &elt))
    {
      m4_diversion *diversion = (m4_diversion *) elt;
      if (diversion->size || diversion->used)
        {
          if (diversion->size)
            xfprintf (file, "D%d,%d\n", diversion->divnum, diversion->used);
          else
            {
              struct stat file_stat;
              diversion->u.file = m4_tmpopen (diversion->divnum, true);
              if (fstat (fileno (diversion->u.file), &file_stat) < 0)
                M4ERROR ((EXIT_FAILURE, errno, "cannot stat diversion"));
              if (file_stat.st_size < 0
                  || (file_stat.st_size + 0UL
                      != (unsigned long int) file_stat.st_size))
                M4ERROR ((EXIT_FAILURE, 0, "diversion too large"));
              xfprintf (file, "D%d,%lu\n", diversion->divnum,
                        (unsigned long int) file_stat.st_size);
            }

          insert_diversion_helper (diversion);
          putc ('\n', file);

          last_inserted = diversion->divnum;
        }
    }
  gl_oset_iterator_free (&iter);

  /* Save the active diversion number, if not already.  */

  if (saved_number != last_inserted)
    xfprintf (file, "D%d,0\n\n", saved_number);
}
