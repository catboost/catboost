# `appnope`

Simple package for disabling App Nap on macOS >= 10.9,
which can be problematic.

To disable App Nap:

```python
import appnope
appnope.nope()
```

To reenable, for some reason:

```python
appnope.nap()
```

or to only disable App Nap for a particular block:

```
with appnope.nope_scope():
    do_important_stuff()
```

It uses ctypes to wrap a `[NSProcessInfo beginActivityWithOptions]` call to disable App Nap.

To install:

    pip install appnope
