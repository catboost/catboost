### UNU.RAN Source Code for use in SciPy

Modified UNU.RAN source code for use in SciPy. Makefiles, deprecated functions, and unwanted disrectories have been removed.
`unuran/src/urng/urng_default.c` has been replaced by `urng_default_mod.c` which is a modified version of `urng_default.c`
to work with NumPy generators which are set as default on import.

### Update

Run the following script to update UNU.RAN's version (The current version used is 1.8.1):

```shell
python get_and_clean_unuran.py -v --unuran-version 1.8.1
```

Replace the version number after `--unuran-version` to update.
