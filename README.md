# سیستم پرو لباس مجازی بلادرنگ
# Real-Time Virtual Try-On System

این پروژه امکان پیاده‌سازی سیستم پرو لباس مجازی به صورت Real-Time را بدون نیاز به سنسورهای خاص و تنها با یک وبکم و GPU معادل RTX 3060 یا بالاتر فراهم می‌کند. برخلاف سایر روش‌های مبتنی بر تصویر، این متد نیازمند آموزش یک شبکه مجزا برای هر لباس است تا کیفیت و تطابق بهتری برای هر آیتم حاصل شود.

## قابلیت‌ها:
- پردازش و پرو لباس به صورت بلادرنگ با استفاده از شبکه‌های عمیق
- عدم حذف لباس اصلی و صرفاً قرار دادن لباس مجازی روی بدن  
- قابل اجرا روی دسکتاپ و سیستم‌های مجهز به GPU مناسب
- امکان آموزش مدل برای لباس‌های جدید با دیتاست‌های شخصی یا عمومی

## راه‌اندازی پروژه

### گام ۱: نصب وابستگی‌ها
```bash
git clone https://github.com/daneshavaran2/library-finder-app.git
cd library-finder-app
chmod +x setup.sh
./setup.sh
```

یا نصب دستی:
```bash
# نصب وابستگی‌های سیستم (Ubuntu/Debian)
sudo apt install gcc g++ libxcb-xinerama0-dev libxcb-icccm4-dev libxcb-image0-dev libxcb-keysyms1-dev libxcb-randr0-dev libxcb-shape0-dev libxcb-sync-dev libxcb-xfixes0-dev libxcb-xkb-dev 
sudo apt install qtbase5-dev qtbase5-dev-tools libqt5gui5 libqt5widgets5 libqt5multimedia5 libqt5multimediawidgets5 libqt5multimedia5-plugins libpulse-mainloop-glib0

# ایجاد محیط مجازی Python
python3 -m venv venv
source venv/bin/activate

# نصب وابستگی‌های Python
pip install -r requirements.txt
pip install detectron2@git+https://github.com/facebookresearch/detectron2.git
pip install git+https://github.com/ZaiqiangWu/ROMP.git#subdirectory=simple_romp
```

### گام ۲: دانلود مدل‌های آموزش دیده
```bash
sudo apt install git-lfs
git lfs install
./download_models.sh

# یا دانلود دستی از HuggingFace
git clone https://huggingface.co/wuzaiqiang/rtv_ckpts models/rtv_ckpts
```

### گام ۳: تست نصب
```bash
python test_installation.py
```

### گام ۴: اجرای دموی بلادرنگ
اطمینان حاصل کنید که وبکم متصل است.
```bash
python rtl_demo.py
```

برای گزینه‌های بیشتر:
```bash
python rtl_demo.py --help
python rtl_demo.py --device cuda --camera-id 0 --verbose
```

## آموزش مدل برای لباس جدید
برای آموزش مدل جدید برای آیتم لباس، به [راهنمای آموزش](./training_instructions.md) مراجعه کنید و دیتاست مناسب را آماده و شبکه را آموزش دهید.

## نکات توسعه‌دهندگان
- تمام کدها مبتنی بر پایتون ۳.۹ و کتابخانه‌های PyTorch و OpenCV هستند
- برای افزودن قابلیت‌های جدید (مانند بهبود شناسایی بدن یا افکت‌های گرافیکی) کافی است ماژول‌های مربوطه را توسعه دهید
- مدل‌های شناسایی بدن (ROMP) و تشخیص لباس (Detectron2) به صورت ماژولار قابل جایگزینی یا ارتقاء هستند

## معماری سیستم

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│    Webcam       │───▶│  Body Detection  │───▶│  Pose Estimation│
│    Input        │    │     (ROMP)       │    │   & Keypoints   │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                ▲                        ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Virtual       │◀───│  Garment Overlay │◀───│ Clothing        │
│   Try-On        │    │     Network      │    │ Detection       │
│   Output        │    │    (RTV Model)   │    │ (Detectron2)    │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## ساختار پروژه

```
library-finder-app/
├── rtl_demo.py              # اسکریپت اصلی دمو
├── setup.sh                 # اسکریپت نصب
├── requirements.txt         # وابستگی‌های Python
├── training_instructions.md # راهنمای آموزش
├── test_installation.py     # تست نصب
├── models/                  # مدل‌های آموزش دیده
├── data/                    # دیتاست‌ها
├── checkpoints/             # checkpoint های آموزش
├── config/                  # فایل‌های پیکربندی
├── scripts/                 # اسکریپت‌های کمکی
└── logs/                    # لاگ‌های آموزش
```

## سیستم مورد نیاز

### حداقل سیستم مورد نیاز:
- **GPU**: NVIDIA RTX 3060 یا معادل (8GB VRAM)
- **CPU**: Intel i5-8400 یا AMD Ryzen 5 2600
- **RAM**: 16GB
- **Storage**: 10GB فضای خالی
- **OS**: Ubuntu 18.04+, Windows 10+, macOS 10.15+

### توصیه شده:
- **GPU**: NVIDIA RTX 4070 یا بالاتر (12GB+ VRAM)
- **CPU**: Intel i7-10700K یا AMD Ryzen 7 3700X
- **RAM**: 32GB
- **Storage**: 50GB SSD

## عملکرد

| GPU Model | Resolution | FPS | Quality |
|-----------|------------|-----|---------|
| RTX 3060  | 640x480   | 25-30 | خوب    |
| RTX 3070  | 1024x768  | 30-35 | عالی   |
| RTX 4070  | 1920x1080 | 40-45 | عالی   |
| RTX 4080  | 1920x1080 | 50-60 | فوق‌العاده |

## مشارکت در پروژه

1. Fork کنید
2. Branch جدید بسازید (`git checkout -b feature/AmazingFeature`)
3. تغییرات را commit کنید (`git commit -m 'Add some AmazingFeature'`)
4. Push کنید (`git push origin feature/AmazingFeature`)
5. Pull Request باز کنید

## مجوز
این پروژه تحت مجوز MIT منتشر شده است. برای جزئیات بیشتر فایل [LICENSE](LICENSE) را مطالعه کنید.

## استناد علمی
```text
@misc{wu2025realtime,
    title={Real-Time Per-Garment Virtual Try-On with Temporal Consistency for Loose-Fitting Garments},
    author={Zaiqiang Wu and I-Chao Shen and Takeo Igarashi},
    year={2025},
    eprint={2506.12348},
    archivePrefix={arXiv},
    primaryClass={cs.GR}
}

@misc{wu2025lowbarrier,
    title={Low-Barrier Dataset Collection with Real Human Body for Interactive Per-Garment Virtual Try-On},
    author={Zaiqiang Wu and Yechen Li and Jingyuan Liu and Yuki Shibata and Takayuki Hori and I-Chao Shen and Takeo Igarashi},
    year={2025},
    eprint={2506.10468},
    archivePrefix={arXiv},
    primaryClass={cs.GR}
}
```

## پشتیبانی و کمک

- **مسائل**: [GitHub Issues](https://github.com/daneshavaran2/library-finder-app/issues)
- **بحث**: [GitHub Discussions](https://github.com/daneshavaran2/library-finder-app/discussions)  
- **ویکی**: [پروژه Wiki](https://github.com/daneshavaran2/library-finder-app/wiki)

## تاریخچه تغییرات

### نسخه 1.0.0 (2025-01-XX)
- پیاده‌سازی اولیه سیستم RTV
- پشتیبانی از real-time processing
- ادغام ROMP و Detectron2
- رابط کاربری ساده برای دمو

---

**توجه**: این پروژه در حال توسعه است و ممکن است تغییرات قابل توجهی در نسخه‌های آتی داشته باشد.