@echo off
echo ===================================================
echo Creating Hospital Readmission Project Structure
echo ===================================================

:: 1. Create the directories
echo Creating folders...
mkdir data
mkdir src

:: 2. Create the root level files
echo Creating root files...
type nul > main.py
type nul > requirements.txt

:: 3. Create the source directory module files
echo Creating src files...
type nul > src\__init__.py
type nul > src\data_processor.py
type nul > src\pipeline_builder.py
type nul > src\model_evaluator.py

echo ===================================================
echo Project structure successfully created!
echo ===================================================
pause