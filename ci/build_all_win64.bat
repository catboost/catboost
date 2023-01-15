@echo on

set WIN_COMMON_FLAGS=-k -DOS_SDK=local -DCUDA_ROOT="%CUDA_PATH%" -DUSE_ARCADIA_CUDA_HOST_COMPILER=no --host-platform-flag USE_ARCADIA_CUDA_HOST_COMPILER=no

call "%VS_VARS_PATH%\vcvars64.bat" -vcvars_ver=14.16
if %errorlevel% neq 0 exit /b %errorlevel%

c:\Python27\python.exe ya make -r -DNO_DEBUGINFO %WIN_COMMON_FLAGS% -DHAVE_CUDA=yes -o . catboost\app
if %errorlevel% neq 0 exit /b %errorlevel%

c:\Python27\python.exe ya make -r -DNO_DEBUGINFO %WIN_COMMON_FLAGS% -DHAVE_CUDA=yes -o . catboost\libs\model_interface\
if %errorlevel% neq 0 exit /b %errorlevel%

cd catboost\python-package
if %errorlevel% neq 0 exit /b %errorlevel%

set PyV=27
echo c:\Python%PyV%\python.exe
c:\Python%PyV%\python.exe mk_wheel.py %WIN_COMMON_FLAGS% -DPYTHON_INCLUDE="/I c:/Python%PyV%/include/" -DPYTHON_LIBRARIES="c:/Python%PyV%/libs/python%PyV%.lib"
if %errorlevel% neq 0 exit /b %errorlevel%

set PyV=35
echo c:\Python%PyV%\python.exe
c:\Python%PyV%\python.exe mk_wheel.py %WIN_COMMON_FLAGS% -DPYTHON_INCLUDE="/I c:/Python%PyV%/include/" -DPYTHON_LIBRARIES="c:/Python%PyV%/libs/python%PyV%.lib"
if %errorlevel% neq 0 exit /b %errorlevel%

set PyV=36
echo c:\Python%PyV%\python.exe
c:\Python%PyV%\python.exe mk_wheel.py %WIN_COMMON_FLAGS% -DPYTHON_INCLUDE="/I c:/Python%PyV%/include/" -DPYTHON_LIBRARIES="c:/Python%PyV%/libs/python%PyV%.lib"
if %errorlevel% neq 0 exit /b %errorlevel%

set PyV=37
echo c:\Python%PyV%\python.exe
c:\Python%PyV%\python.exe mk_wheel.py %WIN_COMMON_FLAGS% -DPYTHON_INCLUDE="/I c:/Python%PyV%/include/" -DPYTHON_LIBRARIES="c:/Python%PyV%/libs/python%PyV%.lib"
if %errorlevel% neq 0 exit /b %errorlevel%

set PyV=38
echo c:\Python%PyV%\python.exe
c:\Python%PyV%\python.exe mk_wheel.py %WIN_COMMON_FLAGS% -DPYTHON_INCLUDE="/I c:/Python%PyV%/include/" -DPYTHON_LIBRARIES="c:/Python%PyV%/libs/python%PyV%.lib"
if %errorlevel% neq 0 exit /b %errorlevel%

echo Building R-package

cd ..\..
call ya.bat make -r -DNO_DEBUGINFO -T %WIN_COMMON_FLAGS% -o . .\catboost\R-package\src\
if %errorlevel% neq 0 exit /b %errorlevel%
cd catboost\R-package
if %errorlevel% neq 0 exit /b %errorlevel%
mkdir catboost


copy DESCRIPTION catboost
if %errorlevel% neq 0 exit /b %errorlevel%
copy NAMESPACE catboost
if %errorlevel% neq 0 exit /b %errorlevel%
copy README.md catboost
if %errorlevel% neq 0 exit /b %errorlevel%

xcopy R catboost\R /S /E /I
if %errorlevel% neq 0 exit /b %errorlevel%
xcopy inst catboost\inst /S /E /I
if %errorlevel% neq 0 exit /b %errorlevel%
xcopy man catboost\man /S /E /I
if %errorlevel% neq 0 exit /b %errorlevel%
xcopy tests catboost\tests /S /E /I
if %errorlevel% neq 0 exit /b %errorlevel%

mkdir catboost\inst\libs
if %errorlevel% neq 0 exit /b %errorlevel%
mkdir catboost\inst\libs\x64
if %errorlevel% neq 0 exit /b %errorlevel%
copy src\libcatboostr.dll catboost\inst\libs\x64
if %errorlevel% neq 0 exit /b %errorlevel%
7z -ttar a dummy catboost -so | 7z -si -tgzip a catboost-R-Windows.tgz
if %errorlevel% neq 0 exit /b %errorlevel%
