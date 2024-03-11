You can create a MS Windows(R) DLL using Cygwin[1] and MS Visual
Studio 2005 [2]:

1. Edit path variables 'VSCYG' and 'VCWIN' in file 
   './scripts/win32/build.sh'
   (The distributed files contains the default pathes of 
   Cygwin and MS Visual Studio).

2. Run './scripts/win32/build.sh'

3. The created files can be then found in directory 
   'unuran-win32':

	libunuran.dll
	libunuran.dll.manifest
	libunuran.def
	libunuran.lib
	libunuran.exp
	unuran.h

4. Applications also have to be linked against the runtime library
   'msvcr80.dll' (included in the MS Visual Studio tree).


References:

[1] http://www.cygwin.com/
[2] http://msdn2.microsoft.com/en-us/vstudio/default.aspx