#!/bin/bash
set -e

echo "=== LLM_From_ZERO 环境安装脚本 ==="

# 检查 Python 版本
PYTHON_VERSION=$(python3 --version 2>&1 | cut -d' ' -f2 | cut -d'.' -f1,2)
if [[ $(echo "$PYTHON_VERSION < 3.10" | bc -l 2>/dev/null || echo 0) -eq 1 ]]; then
    echo "错误: 需要 Python 3.10+, 当前版本 $PYTHON_VERSION"
    exit 1
fi

# 创建虚拟环境
if [ ! -d "venv" ]; then
    echo "创建虚拟环境..."
    python3 -m venv venv
fi

# 激活虚拟环境
source venv/bin/activate

# 升级 pip
echo "升级 pip..."
pip install --upgrade pip

# 安装依赖
echo "安装依赖..."
pip install -r requirements.txt

echo ""
echo "=== 安装完成 ==="
echo "运行以下命令激活环境:"
echo "  source venv/bin/activate"
echo ""
echo "快速开始:"
echo "  jupyter notebook notebooks/01_BPE_Tokenizer.ipynb"