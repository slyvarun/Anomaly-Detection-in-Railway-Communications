@echo off
REM Save the current directory
set current_dir=%cd%

REM Navigate to the folder containing the virtual environment
cd /d C:\Users\ASUS\yolovenv

REM Activate the virtual environment
call scripts\activate

REM Return to the original directory (PWD)
cd /d %current_dir%

REM Keep the command prompt open after activation
cmd
jupyter notebook
pause