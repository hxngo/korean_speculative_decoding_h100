#!/bin/bash
# setup_conda_env.sh - H100 콘다 환경 자동 설정

set -e  # 에러 발생시 중단

echo "🐍 H100 Korean Speculative Decoding - Conda Environment Setup"
echo "================================================================"

# 현재 디렉토리 확인
if [ ! -f "environment.yml" ]; then
    echo "❌ environment.yml file not found in current directory"
    echo "Please run this script from the project root directory"
    exit 1
fi

# 콘다 설치 확인
if ! command -v conda &> /dev/null; then
    echo "❌ Conda is not installed or not in PATH"
    echo "Please install Anaconda or Miniconda first"
    exit 1
fi

echo "✅ Conda found: $(conda --version)"

# 기존 환경 확인 및 제거 (선택사항)
ENV_NAME="h100_korean_spec"
if conda env list | grep -q "^${ENV_NAME}"; then
    echo "⚠️  Environment '${ENV_NAME}' already exists"
    read -p "Do you want to remove and recreate it? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "🗑️  Removing existing environment..."
        conda env remove -n ${ENV_NAME} -y
    else
        echo "ℹ️  Using existing environment"
        conda activate ${ENV_NAME}
        exit 0
    fi
fi

# 환경 생성
echo "🏗️  Creating conda environment from environment.yml..."
conda env create -f environment.yml

# 환경 활성화
echo "🔄 Activating environment..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate ${ENV_NAME}

# CUDA 설치 확인
echo "🔍 Checking CUDA installation..."
python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'Device count: {torch.cuda.device_count()}')
    print(f'Current device: {torch.cuda.current_device()}')
    print(f'Device name: {torch.cuda.get_device_name(0)}')
else:
    print('⚠️  CUDA not available - check your installation')
"

# Transformers 설치 확인
echo "🤗 Checking Transformers installation..."
python -c "
import transformers
print(f'Transformers version: {transformers.__version__}')
"

# 한국어 처리 라이브러리 확인
echo "🇰🇷 Checking Korean language processing..."
python -c "
try:
    import konlpy
    print('✅ KoNLPy installed successfully')
except ImportError:
    print('⚠️  KoNLPy not available')

try:
    import sentencepiece
    print('✅ SentencePiece installed successfully')
except ImportError:
    print('⚠️  SentencePiece not available')
"

# Jupyter 커널 등록
echo "📓 Setting up Jupyter kernel..."
python -m ipykernel install --user --name=${ENV_NAME} --display-name="H100 Korean Spec"

# 환경 정보 저장
echo "📋 Saving environment information..."
conda env export > environment_actual.yml
pip freeze > requirements_actual.txt

echo ""
echo "🎉 Environment setup completed successfully!"
echo "================================================================"
echo "📋 Quick Start Commands:"
echo ""
echo "1. Activate environment:"
echo "   conda activate ${ENV_NAME}"
echo ""
echo "2. Deactivate environment:"
echo "   conda deactivate"
echo ""
echo "3. Start Jupyter:"
echo "   jupyter notebook"
echo ""
echo "4. Download models:"
echo "   python scripts/setup/download_models.py --list"
echo ""
echo "5. Run first experiment:"
echo "   python scripts/experiments/run_m3_reproduction.py"
echo ""
echo "💡 Environment name: ${ENV_NAME}"
echo "📁 Project directory: $(pwd)"
echo "================================================================"