@echo off
rem Builds the DLSS bridge (build\swdlss.dll) with MSVC against NVIDIA's DLSS SDK, and puts the SDK's nvngx_dlss.dll
rem next to it. The SDK is not part of this repository (its license keeps it out of open-source code): get it from
rem https://github.com/NVIDIA/DLSS and pass its folder.
rem usage: tools\build_dlss_bridge.bat <DLSS SDK folder> [Vulkan SDK folder]
setlocal
set SDK=%~1
if "%SDK%"=="" (echo usage: %0 ^<DLSS SDK folder^> [Vulkan SDK folder] & exit /b 1)
set VKSDK=%~2
if "%VKSDK%"=="" for /d %%d in (C:\VulkanSDK\*) do set VKSDK=%%d
set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
for /f "usebackq tokens=*" %%i in (`call "%%VSWHERE%%" -latest -products * -property installationPath`) do set "VS=%%i"
call "%VS%\VC\Auxiliary\Build\vcvars64.bat" >nul || exit /b 1
cd /d "%~dp0.."
if not exist build mkdir build
cl /nologo /LD /O2 /MD /EHsc /std:c++17 /I"%SDK%\include" /I"%VKSDK%\Include" /Iruntime\nvn\vulkan\dlss ^
   runtime\nvn\vulkan\dlss\swdlss.cpp /Fobuild\swdlss.obj /Febuild\swdlss.dll ^
   /link "%SDK%\lib\Windows_x86_64\x64\nvsdk_ngx_d.lib" advapi32.lib user32.lib || exit /b 1
copy /y "%SDK%\lib\Windows_x86_64\rel\nvngx_dlss.dll" build\ >nul || exit /b 1
echo built build\swdlss.dll
