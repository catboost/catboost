#ifndef NUMPY_CORE_SRC_MULTIARRAY_DATETIME_STRINGS_H_
#define NUMPY_CORE_SRC_MULTIARRAY_DATETIME_STRINGS_H_

/*
 * Parses (almost) standard ISO 8601 date strings. The differences are:
 *
 * + The date "20100312" is parsed as the year 20100312, not as
 *   equivalent to "2010-03-12". The '-' in the dates are not optional.
 * + Only seconds may have a decimal point, with up to 18 digits after it
 *   (maximum attoseconds precision).
 * + Either a 'T' as in ISO 8601 or a ' ' may be used to separate
 *   the date and the time. Both are treated equivalently.
 * + Doesn't (yet) handle the "YYYY-DDD" or "YYYY-Www" formats.
 * + Doesn't handle leap seconds (seconds value has 60 in these cases).
 * + Doesn't handle 24:00:00 as synonym for midnight (00:00:00) tomorrow
 * + Accepts special values "NaT" (not a time), "Today", (current
 *   day according to local time) and "Now" (current time in UTC).
 *
 * 'str' must be a NULL-terminated string, and 'len' must be its length.
 * 'unit' should contain -1 if the unit is unknown, or the unit
 *      which will be used if it is.
 * 'casting' controls how the detected unit from the string is allowed
 *           to be cast to the 'unit' parameter.
 *
 * 'out' gets filled with the parsed date-time.
 * 'out_bestunit' gives a suggested unit based on the amount of
 *      resolution provided in the string, or -1 for NaT.
 * 'out_special' gets set to 1 if the parsed time was 'today',
 *      'now', or ''/'NaT'. For 'today', the unit recommended is
 *      'D', for 'now', the unit recommended is 's', and for 'NaT'
 *      the unit recommended is 'Y'.
 *
 * Returns 0 on success, -1 on failure.
 */
NPY_NO_EXPORT int
parse_iso_8601_datetime(char const *str, Py_ssize_t len,
                    NPY_DATETIMEUNIT unit,
                    NPY_CASTING casting,
                    npy_datetimestruct *out,
                    NPY_DATETIMEUNIT *out_bestunit,
                    npy_bool *out_special);

/*
 * Provides a string length to use for converting datetime
 * objects with the given local and unit settings.
 */
NPY_NO_EXPORT int
get_datetime_iso_8601_strlen(int local, NPY_DATETIMEUNIT base);

/*
 * Converts an npy_datetimestruct to an (almost) ISO 8601
 * NULL-terminated string.
 *
 * If 'local' is non-zero, it produces a string in local time with
 * a +-#### timezone offset, otherwise it uses timezone Z (UTC).
 *
 * 'base' restricts the output to that unit. Set 'base' to
 * -1 to auto-detect a base after which all the values are zero.
 *
 *  'tzoffset' is used if 'local' is enabled, and 'tzoffset' is
 *  set to a value other than -1. This is a manual override for
 *  the local time zone to use, as an offset in minutes.
 *
 *  'casting' controls whether data loss is allowed by truncating
 *  the data to a coarser unit. This interacts with 'local', slightly,
 *  in order to form a date unit string as a local time, the casting
 *  must be unsafe.
 *
 *  Returns 0 on success, -1 on failure (for example if the output
 *  string was too short).
 */
NPY_NO_EXPORT int
make_iso_8601_datetime(npy_datetimestruct *dts, char *outstr, npy_intp outlen,
                    int local, int utc, NPY_DATETIMEUNIT base, int tzoffset,
                    NPY_CASTING casting);

/*
 * This is the Python-exposed datetime_as_string function.
 */
NPY_NO_EXPORT PyObject *
array_datetime_as_string(PyObject *NPY_UNUSED(self), PyObject *args,
                                PyObject *kwds);

#endif  /* NUMPY_CORE_SRC_MULTIARRAY_DATETIME_STRINGS_H_ */
