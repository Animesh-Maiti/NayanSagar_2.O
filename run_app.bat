@echo off
cd /d "%~dp0"
call .\.venv\Scripts\activate.bat
streamlit run app/main.py --server.headless true --server.port 8501
