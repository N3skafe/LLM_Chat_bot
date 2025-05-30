@echo off
setlocal
cd /d "%~dp0"
if not exist "pyproject.toml" (
    echo Run this from the m_agent_chatbot folder ^(project root^).
    exit /b 1
)
poetry run python -m streamlit run src\multi_agent_chatbot\main.py
