REM @echo off

@if '%1' == '' goto HOME
@if '%1' == 'HOME' goto HOME
@if '%1' == 'OFFICE' goto OFFICE

goto OFFICE

:OFFICE
set TRUSTED_HOSTS=--trusted-host pypi.org --trusted-host files.pythonhosted.org --trusted-host wrepp0401.cpr.ca
python.exe -m pip install --upgrade pip
pip install %TRUSTED_HOSTS% --upgrade certifi
pip install %TRUSTED_HOSTS% -r requirements.txt

goto END

:HOME
python.exe -m pip install --upgrade pip
pip install  --upgrade certifi
pip install -r requirements.txt
goto END


:END