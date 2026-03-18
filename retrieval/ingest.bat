@echo off
REM Ingest corpus data into both Qdrant Cloud collections.
REM Run from the repository root:  retrieval\ingest.bat

if "%CSV_PATH%"=="" set CSV_PATH=data\corpus.csv
if "%BATCH_SIZE%"=="" set BATCH_SIZE=64

REM Auto-detect GPU via nvidia-smi; Python scripts fall back to CPU automatically
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader >nul 2>&1
if %ERRORLEVEL% equ 0 (
    for /f "tokens=*" %%G in ('nvidia-smi --query-gpu^=name^,memory.total --format^=csv^,noheader 2^>nul') do (
        echo [Device] GPU detected: %%G
        if "%CUDA_DEVICE%"=="" set CUDA_DEVICE=0
        set CUDA_VISIBLE_DEVICES=%CUDA_DEVICE%
        goto :config
    )
)
echo [Device] No GPU detected - running on CPU (ingestion will be slow)
:config
echo CSV:         %CSV_PATH%
echo Batch size:  %BATCH_SIZE%
echo ========================

cd /d "%~dp0"

echo.
echo ^>^>^> [1/2] Ingesting into BGE-m3 collection...
python ingest_bge.py --csv "..\%CSV_PATH%" --batch-size %BATCH_SIZE%
if %ERRORLEVEL% neq 0 (
    echo ERROR: BGE ingestion failed.
    exit /b 1
)

echo.
echo ^>^>^> [2/2] Ingesting into E5 collection...
python ingest_e5.py --csv "..\%CSV_PATH%" --batch-size %BATCH_SIZE%
if %ERRORLEVEL% neq 0 (
    echo ERROR: E5 ingestion failed.
    exit /b 1
)

echo.
echo === All ingestion complete ===
