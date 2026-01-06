@echo off
chcp 65001 > nul

echo ========================================
echo 医疗智能问答系统启动程序
echo ========================================

REM 设置Python虚拟环境路径（根据您的实际路径修改）
set VENV_PATH=D:\PycharmProjects\exp04-easy-rag-system\.venv\Scripts

REM 激活虚拟环境
if exist "%VENV_PATH%\activate.bat" (
    call "%VENV_PATH%\activate.bat"
) else (
    echo 警告：未找到虚拟环境
    echo 将使用系统Python环境
)

REM 设置环境变量（强制离线模式）
set TRANSFORMERS_OFFLINE=1
set HF_HUB_OFFLINE=1
set HF_ENDPOINT=https://hf-mirror.com
set HF_HOME=D:\.cache\huggingface
set TRANSFORMERS_CACHE=D:\.cache\huggingface

echo ✅ 环境变量已设置
echo 🔒 离线模式：已启用
echo 📁 缓存路径：%HF_HOME%

REM 检查数据文件
if not exist "data\processed_data.json" (
    echo ⚠️ 警告：未找到数据文件
    echo 💡 请运行：python preprocess.py
    pause
    exit /b 1
)

REM 检查依赖
echo.
echo 🔍 检查系统依赖...
where streamlit >nul 2>nul
if errorlevel 1 (
    echo ❌ 未找到streamlit，请安装：pip install streamlit
    pause
    exit /b 1
)

REM 启动应用
echo.
echo 🚀 正在启动Streamlit应用...
echo 🌐 请访问：http://localhost:8501
echo ⏳ 启动中...（首次启动可能需要几分钟）

streamlit run app.py

pause