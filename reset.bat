@echo off

setlocal
:PROMPT
SET /P AREYOUSURE=确定重置么 (Y/[N])?
IF /I "%AREYOUSURE%" NEQ "Y" GOTO END

echo 重置开始
git reset --hard HEAD~1
echo 重置结束

:END
endlocal
pause
