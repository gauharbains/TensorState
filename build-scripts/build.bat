call py35build.bat
call py36build.bat
call py37build.bat
call py38build.bat

python -m twine upload --repository pypi dist/* -u %TWINE_PASS% -p %TWINE_PASS%