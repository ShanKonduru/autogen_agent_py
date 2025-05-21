@echo off
set TRUSTED_HOSTS=--trusted-host pypi.org --trusted-host files.pythonhosted.org --trusted-host wrepp0401.cpr.ca

python.exe -m pip install --upgrade pip
pip install %TRUSTED_HOSTS% --upgrade certifi
pip install %TRUSTED_HOSTS% -r requirements.txt
