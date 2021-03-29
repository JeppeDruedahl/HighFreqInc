@echo off
start %windir%\System32\cmd.exe "/K" taskkill /F /IM sas.exe /T
exit