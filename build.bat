@echo off
g++ main.cpp image_operation_functions.cpp -o out -lOpenCL -mavx512f
if %errorlevel% neq 0 (
    echo Build failed with error code %errorlevel%.
    pause
    exit /b %errorlevel%
)
echo Build succeeded.