@echo off
g++ main.cpp cpu_img_ops.cpp -o default
if %errorlevel% neq 0 (
    echo Build failed with error code %errorlevel%.
    pause
    exit /b %errorlevel%
)
echo Build succeeded.