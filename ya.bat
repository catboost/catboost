@echo off
rem Ya Simple Windows launcher
setlocal
call :dbg Ya Simple Windows Launcher (Debug)
call :find_ya
if ERRORLEVEL 1 exit /b 1
call :dbg Ya: %YA_BAT_REAL%
call :find_python
if ERRORLEVEL 1 exit /b 1
call :dbg Python: "%YA_BAT_PYTHON%"
call "%YA_BAT_PYTHON%" "%YA_BAT_REAL%" %*
exit /b %ERRORLEVEL%

:find_ya
call :dbg Searching for ya near ya.bat...
set YA_BAT_REAL=%~dp0ya
if exist "%YA_BAT_REAL%" exit /b 0
call :err Ya not found
exit /b 1

:find_python
call :dbg Searching for python in PATH...
for /f "delims=" %%F in ('where python 2^>nul') do (
    call :test_python %%~sF
    if not ERRORLEVEL 1 (
        set YA_BAT_PYTHON=%%F
        exit /b 0
    )
)
call :dbg Searching for python in ftypes...
for /f delims^=^=^"^ tokens^=2 %%F in ('ftype Python.File 2^>nul') do (
    call :test_python %%F
    if not ERRORLEVEL 1 (
        set YA_BAT_PYTHON=%%F
        exit /b 0
    )
)
call :dbg Searching for python manually...
for %%F in (
    C:\Python27\python.exe
    C:\Python26\python.exe
) do (
    call :test_python %%F
    if not ERRORLEVEL 1 (
        set YA_BAT_PYTHON=%%F
        exit /b 0
    )
)
call :err Python not found
exit /b 1

:test_python
call :dbg -- Checking python: %1
if not exist %1 (
    call :dbg ---- Not found
    exit /b 1
)
for /f %%P in ('%1 -c "import os, sys; sys.stdout.write(os.name + '\n')" 2^>nul') do set YA_BAT_PYTHON_PLATFORM=%%P
if not defined YA_BAT_PYTHON_PLATFORM (
    call :dbg ---- Not runnable
    exit /b 2
)
if not "%YA_BAT_PYTHON_PLATFORM%"=="nt" (
    call :dbg ---- Non-windows: %YA_BAT_PYTHON_PLATFORM%
    exit /b 3
)
exit /b 0

:dbg
if defined YA_BAT_DEBUG echo [ya.bat] %* 1>&2
exit /b 0

:err
echo [ya.bat] Error: %* 1>&2
exit /b 0
