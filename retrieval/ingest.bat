@echo off
REM Ingest corpus data into both Qdrant Cloud collections.
REM Run from the repository root:  retrieval\ingest.bat

if "%CSV_PATH%"=="" set CSV_PATH=data\corpus.csv
if "%BATCH_SIZE%"=="" set BATCH_SIZE=64

echo === Ingestion config ===
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
