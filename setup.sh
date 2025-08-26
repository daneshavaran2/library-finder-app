#!/bin/bash

# راهنمای نصب سیستم پرو لباس مجازی بلادرنگ
# Real-Time Virtual Try-On Installation Script

set -e

echo "=== شروع نصب سیستم پرو لباس مجازی بلادرنگ ==="
echo "=== Starting Real-Time Virtual Try-On Installation ==="

# بررسی پایتون
echo "بررسی نسخه پایتون..."
python_version=$(python3 --version 2>&1 | grep -o '[0-9]\+\.[0-9]\+' | head -1)
echo "Python version: $python_version"

if ! python3 -c "import sys; assert sys.version_info >= (3, 8)"; then
    echo "خطا: پایتون 3.8 یا بالاتر مورد نیاز است"
    echo "Error: Python 3.8 or higher is required"
    exit 1
fi

# بررسی CUDA (اختیاری)
echo "بررسی پشتیبانی CUDA..."
if command -v nvidia-smi &> /dev/null; then
    echo "NVIDIA GPU detected:"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
    echo "CUDA support available"
else
    echo "هشدار: GPU NVIDIA یافت نشد. سیستم روی CPU اجرا خواهد شد"
    echo "Warning: No NVIDIA GPU found. System will run on CPU"
fi

# نصب وابستگی‌های سیستم
echo "نصب وابستگی‌های سیستم..."
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    # Ubuntu/Debian
    if command -v apt &> /dev/null; then
        echo "Installing system dependencies for Ubuntu/Debian..."
        sudo apt update
        sudo apt install -y \
            gcc g++ \
            libxcb-xinerama0-dev libxcb-icccm4-dev libxcb-image0-dev \
            libxcb-keysyms1-dev libxcb-randr0-dev libxcb-shape0-dev \
            libxcb-sync-dev libxcb-xfixes0-dev libxcb-xkb-dev \
            qtbase5-dev qtbase5-dev-tools \
            libqt5gui5 libqt5widgets5 libqt5multimedia5 \
            libqt5multimediawidgets5 libqt5multimedia5-plugins \
            libpulse-mainloop-glib0 \
            git git-lfs \
            python3-pip python3-venv \
            libopencv-dev python3-opencv
    elif command -v yum &> /dev/null; then
        # CentOS/RHEL
        echo "Installing system dependencies for CentOS/RHEL..."
        sudo yum install -y \
            gcc gcc-c++ \
            qt5-qtbase-devel qt5-qtmultimedia-devel \
            opencv-devel python3-opencv \
            git git-lfs \
            python3-pip
    fi
elif [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS
    echo "Installing system dependencies for macOS..."
    if command -v brew &> /dev/null; then
        brew install qt5 opencv git git-lfs python3
    else
        echo "لطفا ابتدا Homebrew را نصب کنید: https://brew.sh"
        echo "Please install Homebrew first: https://brew.sh"
        exit 1
    fi
fi

# ایجاد محیط مجازی پایتون
echo "ایجاد محیط مجازی پایتون..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
fi

# فعال‌سازی محیط مجازی
echo "فعال‌سازی محیط مجازی..."
source venv/bin/activate

# ارتقاء pip
echo "ارتقاء pip..."
pip install --upgrade pip setuptools wheel

# نصب وابستگی‌های پایتون
echo "نصب وابستگی‌های پایتون..."
pip install -r requirements.txt

# نصب PyTorch با پشتیبانی CUDA (در صورت وجود)
echo "نصب PyTorch..."
if command -v nvidia-smi &> /dev/null; then
    # نصب PyTorch با CUDA
    echo "Installing PyTorch with CUDA support..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
else
    # نصب PyTorch برای CPU
    echo "Installing PyTorch for CPU..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
fi

# نصب Detectron2
echo "نصب Detectron2..."
pip install 'git+https://github.com/facebookresearch/detectron2.git'

# نصب ROMP
echo "نصب ROMP..."
pip install 'git+https://github.com/ZaiqiangWu/ROMP.git#subdirectory=simple_romp'

# ایجاد دایرکتوری مدل‌ها
echo "ایجاد دایرکتوری مدل‌ها..."
mkdir -p models checkpoints data

# راه‌اندازی Git LFS
echo "راه‌اندازی Git LFS..."
git lfs install

# ایجاد اسکریپت دانلود مدل‌ها
cat > download_models.sh << 'EOF'
#!/bin/bash
# اسکریپت دانلود مدل‌های آموزش دیده

echo "دانلود مدل‌های آموزش دیده..."
echo "Downloading pre-trained models..."

# ایجاد دایرکتوری مدل‌ها
mkdir -p models/rtv_ckpts

# دانلود مدل‌ها از HuggingFace (نمونه)
echo "توجه: برای دانلود مدل‌های واقعی، به مخزن HuggingFace مراجعه کنید"
echo "Note: For actual model download, refer to the HuggingFace repository"
echo "https://huggingface.co/wuzaiqiang/rtv_ckpts"

# در حال حاضر، فایل‌های نمونه ایجاد می‌شوند
touch models/rtv_ckpts/model_checkpoint.pth
touch models/rtv_ckpts/config.yaml

echo "مدل‌های نمونه ایجاد شدند. برای استفاده واقعی، مدل‌های آموزش دیده را دانلود کنید."
echo "Sample model files created. For actual use, download the trained models."
EOF

chmod +x download_models.sh

# ایجاد اسکریپت تست
cat > test_installation.py << 'EOF'
#!/usr/bin/env python3
"""
تست نصب سیستم پرو لباس مجازی
Installation Test Script
"""

import sys
import importlib
import subprocess

def test_import(module_name, package_name=None):
    """تست import کتابخانه"""
    try:
        importlib.import_module(module_name)
        print(f"✓ {package_name or module_name}")
        return True
    except ImportError as e:
        print(f"✗ {package_name or module_name}: {e}")
        return False

def test_gpu():
    """تست پشتیبانی GPU"""
    try:
        import torch
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0)
            print(f"✓ CUDA available: {gpu_count} GPU(s) - {gpu_name}")
            return True
        else:
            print("⚠ CUDA not available - will use CPU")
            return False
    except Exception as e:
        print(f"✗ GPU test failed: {e}")
        return False

def test_camera():
    """تست دوربین"""
    try:
        import cv2
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            print("✓ Camera accessible")
            cap.release()
            return True
        else:
            print("⚠ Camera not accessible")
            return False
    except Exception as e:
        print(f"✗ Camera test failed: {e}")
        return False

def main():
    print("=== تست نصب سیستم پرو لباس مجازی ===")
    print("=== RTV System Installation Test ===\n")
    
    # تست کتابخانه‌های اصلی
    print("Testing core libraries:")
    modules_ok = []
    modules_ok.append(test_import("torch", "PyTorch"))
    modules_ok.append(test_import("torchvision"))
    modules_ok.append(test_import("cv2", "OpenCV"))
    modules_ok.append(test_import("numpy"))
    modules_ok.append(test_import("PIL", "Pillow"))
    
    print("\nTesting optional libraries:")
    test_import("detectron2")
    test_import("simple_romp", "ROMP")
    
    # تست GPU
    print("\nTesting GPU support:")
    gpu_ok = test_gpu()
    
    # تست دوربین
    print("\nTesting camera:")
    camera_ok = test_camera()
    
    # خلاصه
    print("\n=== خلاصه نتایج ===")
    print("=== Summary ===")
    
    if all(modules_ok):
        print("✓ All core modules installed successfully")
    else:
        print("✗ Some core modules missing")
    
    if gpu_ok:
        print("✓ GPU acceleration available")
    else:
        print("⚠ GPU acceleration not available")
    
    if camera_ok:
        print("✓ Camera ready")
    else:
        print("⚠ Camera issues detected")
    
    print("\nInstallation test completed!")
    return 0 if all(modules_ok) else 1

if __name__ == "__main__":
    exit(main())
EOF

chmod +x test_installation.py

echo ""
echo "=== نصب کامل شد! ==="
echo "=== Installation Complete! ==="
echo ""
echo "برای تست نصب:"
echo "To test installation:"
echo "  python test_installation.py"
echo ""
echo "برای دانلود مدل‌ها:"
echo "To download models:"
echo "  ./download_models.sh"
echo ""
echo "برای اجرای دمو:"
echo "To run demo:"
echo "  python rtl_demo.py"
echo ""
echo "برای فعال‌سازی محیط مجازی:"
echo "To activate virtual environment:"
echo "  source venv/bin/activate"
echo ""