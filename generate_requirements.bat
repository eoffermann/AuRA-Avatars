@echo off
REM Generate requirements.txt
pip freeze > requirements.txt

REM Define the custom PyTorch URL
set TORCH_URL=-f https://download.pytorch.org/whl/torch_stable.html

REM Create a temporary file
set TEMP_FILE=requirements_temp.txt

REM Process requirements.txt to append the URL to torch packages
(for /f "usebackq delims=" %%A in (`type requirements.txt`) do (
    echo %%A | findstr /r "^torch== ^torchvision== ^torchaudio==" >nul && (
        echo %%A %TORCH_URL%
    ) || (
        echo %%A
    )
)) > %TEMP_FILE%

REM Replace original requirements.txt with the modified one
move /Y %TEMP_FILE% requirements.txt >nul

REM Clean up and exit
echo Updated requirements.txt with PyTorch URL for torch, torchvision, and torchaudio.
exit /b
