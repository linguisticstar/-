#!/bin/bash
# 医疗RAG系统启动脚本

echo "========================================"
echo "🏥 医疗智能问答系统启动程序"
echo "========================================"

# 设置环境变量（离线模式）
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=~/.cache/huggingface
export TRANSFORMERS_CACHE=~/.cache/huggingface

echo "✅ 环境变量已设置"
echo "🔒 离线模式：已启用"
echo "📁 缓存路径：$HF_HOME"

# 检查必要文件
if [ ! -f "data/processed_data.json" ]; then
    echo "⚠️  警告：未找到数据文件"
    echo "💡 请运行：python preprocess.py"
    exit 1
fi

# 启动应用
echo ""
echo "🚀 正在启动Streamlit应用..."
echo "🌐 请访问：http://localhost:8501"
echo "⏳ 启动中...（首次启动可能需要几分钟）"

streamlit run app.py