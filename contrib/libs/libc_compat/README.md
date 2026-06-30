This library implements a compatibility layer between various libc implementations.

The rationale for the library implementation is described in https://st.yandex-team.ru/IGNIETFERRO-1439.

The code is taken from multiple sources, thus both LICENSE() and VERSION() tags are not very representative.


During development one can make use of the following mapping of `OS_SDK` into glibc version.

| Ubuntu | glibc |
| ------ | ----- |
| 24.04 | 2.39 |
| 22.04 | 2.35 |
| 20.04 | 2.30 |
| 18.04 | 2.27 | 
| 16.04 | 2.23 | 
| 14.04 | 2.18 | 
| 12.04 | 2.15 |
| 10.04 | 2.11 |

Use the following commands to update the table above:

1. `ya make util -DOS_SDK=ubuntu-xx -G | grep OS_SDK_ROOT | head -n 1`
2. `cd ~/.ya/tools/v4/$RESOURCE_ID`
3. `readelf -V $(find . -name 'libc.so.6')`
4. Take the latest version from `.gnu.version_d` section prior to `GLIBC_PRIVATE`
