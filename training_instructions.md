# راهنمای آموزش مدل برای لباس جدید
# Training Guide for New Garment Models

## مقدمه
این راهنما مراحل آموزش مدل جدید برای آیتم‌های لباس در سیستم پرو لباس مجازی بلادرنگ را توضیح می‌دهد.

## پیش‌نیازها
- GPU با حداقل 8GB حافظه (RTX 3060 یا بالاتر توصیه می‌شود)
- Python 3.9+
- نصب کامل وابستگی‌ها طبق راهنمای نصب

## آماده‌سازی دیتاست

### ساختار دیتاست
```
data/
├── garments/
│   ├── shirt_001/
│   │   ├── images/          # تصاویر لباس
│   │   ├── masks/           # ماسک‌های segmentation  
│   │   └── metadata.json    # اطلاعات متا
│   ├── dress_001/
│   └── ...
├── people/
│   ├── person_001/
│   │   ├── images/          # تصاویر شخص
│   │   ├── poses/           # keypoint های بدن
│   │   └── metadata.json
│   └── ...
└── pairs/
    ├── train.txt            # جفت‌های آموزشی
    ├── test.txt             # جفت‌های تست
    └── val.txt              # جفت‌های validation
```

### جمع‌آوری دیتا

#### 1. تصاویر لباس
- کیفیت بالا (حداقل 512x512)
- پس‌زمینه ساده و یکنواخت
- نورپردازی مناسب
- حداقل 100 تصویر از زوایای مختلف

#### 2. تصاویر شخص
- افراد مختلف با اندام‌های متنوع
- پوزهای مختلف
- نورپردازی متنوع
- حداقل 500 تصویر

#### 3. ماسک‌گذاری
```bash
# اجرای اسکریپت ماسک‌گذاری خودکار
python scripts/create_masks.py --input data/garments/shirt_001/images --output data/garments/shirt_001/masks
```

## پیکربندی آموزش

### 1. ایجاد فایل پیکربندی
```yaml
# config/train_config.yaml
model:
  name: "RTVNet"
  backbone: "resnet50"
  pretrained: true
  
dataset:
  garment_path: "data/garments/shirt_001"
  people_path: "data/people"
  image_size: [512, 512]
  batch_size: 8
  
training:
  epochs: 100
  learning_rate: 0.0001
  weight_decay: 0.0001
  scheduler: "cosine"
  
augmentation:
  horizontal_flip: 0.5
  rotation: 10
  color_jitter: 0.2
  
loss:
  perceptual_weight: 1.0
  adversarial_weight: 0.1
  identity_weight: 5.0
```

### 2. اجرای آموزش
```bash
# آموزش مدل جدید
python train.py --config config/train_config.yaml --garment shirt_001

# آموزش با checkpoint موجود
python train.py --config config/train_config.yaml --garment shirt_001 --resume checkpoints/shirt_001/last.pth

# آموزش multi-GPU
python -m torch.distributed.launch --nproc_per_node=2 train.py --config config/train_config.yaml --garment shirt_001
```

## مراحل آموزش

### مرحله 1: Pre-training
```bash
# آموزش مدل پایه روی دیتاست عمومی
python pretrain.py --config config/pretrain_config.yaml
```

### مرحله 2: Fine-tuning
```bash
# fine-tune روی لباس خاص
python finetune.py --config config/train_config.yaml --pretrained models/pretrained_base.pth --garment shirt_001
```

### مرحله 3: Post-processing
```bash
# بهینه‌سازی مدل برای استنتاج
python optimize_model.py --input checkpoints/shirt_001/best.pth --output models/shirt_001_optimized.pth
```

## نظارت بر آموزش

### استفاده از TensorBoard
```bash
# راه‌اندازی TensorBoard
tensorboard --logdir logs/

# مشاهده metrics در مرورگر
# http://localhost:6006
```

### Metrics مهم
- **Perceptual Loss**: کیفیت بصری تصویر
- **Identity Loss**: حفظ هویت شخص
- **Adversarial Loss**: واقعی بودن تصویر
- **SSIM**: شباهت ساختاری
- **FID**: کیفیت کلی تولید

## ارزیابی مدل

### تست خودکار
```bash
# ارزیابی کمی مدل
python evaluate.py --model models/shirt_001_optimized.pth --test_data data/test/

# تولید نمونه‌های تصویری
python generate_samples.py --model models/shirt_001_optimized.pth --num_samples 50
```

### معیارهای ارزیابی
- **SSIM**: > 0.8 (عالی)
- **LPIPS**: < 0.2 (عالی)  
- **FID**: < 50 (خوب)
- **IS**: > 3.0 (خوب)

## بهینه‌سازی برای Real-time

### 1. کوانتیزاسیون
```bash
# تبدیل به INT8 برای سرعت بیشتر
python quantize_model.py --input models/shirt_001_optimized.pth --output models/shirt_001_int8.pth
```

### 2. Pruning
```bash
# حذف وزن‌های غیرضروری
python prune_model.py --input models/shirt_001_optimized.pth --sparsity 0.3 --output models/shirt_001_pruned.pth
```

### 3. TensorRT (برای GPU NVIDIA)
```bash
# بهینه‌سازی با TensorRT
python tensorrt_optimize.py --input models/shirt_001_optimized.pth --output models/shirt_001_tensorrt.trt
```

## استفاده از مدل آموزش دیده

### تست مدل
```bash
# تست مدل جدید
python rtl_demo.py --model models/shirt_001_optimized.pth --garment shirt_001
```

### ادغام در سیستم اصلی
```python
# استفاده در کد
from rtv_system import RealTimeVirtualTryOn

rtv = RealTimeVirtualTryOn(model_path="models/shirt_001_optimized.pth")
rtv.run_real_time_demo()
```

## نکات و ترفندها

### بهبود کیفیت
1. **دیتای بیشتر**: حداقل 1000 جفت train
2. **Augmentation قوی**: تنوع در دیتا
3. **Multi-scale training**: رزولوشن‌های مختلف
4. **Progressive growing**: شروع از رزولوشن پایین

### کاهش زمان آموزش
1. **Mixed precision**: استفاده از FP16
2. **Gradient checkpointing**: کاهش مصرف RAM
3. **DataLoader optimization**: پردازش موازی
4. **Resume training**: ادامه از checkpoint

### عیب‌یابی مشکلات متداول
1. **Overfitting**: Dropout + Regularization
2. **Mode collapse**: تنوع در loss function
3. **Artifacts**: کیفیت دیتا + Architecture
4. **Slow convergence**: Learning rate + Scheduler

## منابع مفید
- [آموزش PyTorch](https://pytorch.org/tutorials/)
- [راهنمای Detectron2](https://detectron2.readthedocs.io/)
- [مستندات ROMP](https://github.com/ZaiqiangWu/ROMP)
- [TensorBoard Guide](https://pytorch.org/docs/stable/tensorboard.html)