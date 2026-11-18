@echo off
:: BatchGotAdmin
:-------------------------------------
REM --> Check for permissions
>nul 2>&1 "%SYSTEMROOT%\system32\cacls.exe" "%SYSTEMROOT%\system32\config\system"

REM --> If error flag set, we do not have admin.
if '%errorlevel%' NEQ '0' (
    echo Requesting administrative privileges...
    goto UACPrompt
) else ( goto gotAdmin )

:UACPrompt
    echo Set UAC = CreateObject^("Shell.Application"^) > "%temp%\getadmin.vbs"
    echo UAC.ShellExecute "cmd.exe", "/c """"%~f0""""", "", "runas", 1 >> "%temp%\getadmin.vbs"
    "%temp%\getadmin.vbs"
    del "%temp%\getadmin.vbs"
    exit /B

:gotAdmin
    pushd "%CD%"
    CD /D "%~dp0"
:--------------------------------------

echo.
echo ========================================================
echo UNBLOCKING PYTHON AND JARVIS PHONE CONNECTION...
echo ========================================================
echo.
echo [1/3] Deleting Block rules for python.exe...
netsh advfirewall firewall delete rule name="python.exe"
netsh advfirewall firewall delete rule name="python"
netsh advfirewall firewall delete rule name="pythonw.exe"
powershell -NoProfile -Command "Get-NetFirewallRule | Where-Object { $_.Action -eq 'Block' -and ($_.DisplayName -match 'python' -or $_.Name -match 'python') } | Remove-NetFirewallRule" 2>nul

echo.
echo [2/3] Adding Allow rule for JARVIS Phone and Python...
netsh advfirewall firewall delete rule name="JARVIS Python Allow" 2>nul
netsh advfirewall firewall add rule name="JARVIS Python Allow" dir=in action=allow program="C:\Users\ACER\AppData\Local\Python\pythoncore-3.14-64\python.exe" profile=any
netsh advfirewall firewall delete rule name="JARVIS phone" 2>nul
netsh advfirewall firewall add rule name="JARVIS phone" dir=in action=allow protocol=TCP localport=8765-8775 profile=any

echo.
echo [3/3] Setting Network to Private...
powershell -NoProfile -Command "Get-NetConnectionProfile | Set-NetConnectionProfile -NetworkCategory Private" 2>nul

echo.
echo ========================================================
echo SUCCESS! All blocks removed and JARVIS allowed.
echo Ab phone me "Retry" karein ya QR code scan karein!
echo ========================================================
echo.
pause
