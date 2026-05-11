# FDConv vs WDConv: Frekans ve Wavelet Dinamik Evrişim

## Proje Yapısı

- **FDConv_WDConv_Classification**: ImageNet-64x64 üzerinde ResNet-18 varyantları kullanarak görüntü sınıflandırması deneyleri
- **FDConv_WDConv_Detection**: PASCAL VOC 2007 üzerinde ResNet-50 backbone'u ile Faster R-CNN kullanarak nesne tespiti deneyleri
- **RESULTS**: Her iki görev için kıyaslama sonuçları, metrikler ve görselleştirmeler

## Notebooks

Colab Workflow:

1. Not defterini Google Colab'a yükleyin
2. GPU runtime seçin (tavsiye)
3. Proje klasörü içerisindeki diğer scriptleri de workspace'e yükleyin (wdconv.py, engine.py, models.py, metrics.py, voc_dataset.py, vb.)

### Sınıflandırma: FDConv_WDConv_Classification/FDConv_Classification.ipynb

ImageNet-64x64 (alt küme) üzerinde üç ResNet-18 varyantını karşılaştırır:
- Temel: Standart Conv2d
- FDConv: Frekans Dinamik Evrişim (CVPR 2025)
- WDConv: Wavelet Dinamik Evrişim

**GPU Runtime:** T4 GPU

**Ana yollar (Hücre 4'te düzenleyin):**
- `DATA_ROOT`: ImageNet-64x64 pickle dosyalarının bulunduğu yer
- `OUTPUT_DIR`: Sonuçların Drive'a kaydedileceği yer

**Veri seti gereksinimleri:**
- Pickle formatında ImageNet-64x64 (deneyde yaklaşık ~150k görüntü kullanılmıştır)
- Dizin yapısı:
  ```
  DATA_ROOT/
    train/
      train_data_batch_1
      train_data_batch_2
      ...
      train_data_batch_10
    val/
      val_data
  ```
- Resmi kaynaktan indirin veya proje yöneticilerinden önceden işlenmiş sürümü kullanın

**Requirements (otomatik olarak yüklenir):**
- torch, torchvision, timm, tqdm, matplotlib, pandas, numpy

### Nesne Tespiti: FDConv_WDConv_Detection/FDConv_Detection.ipynb

PASCAL VOC 2007 üzerinde üç Faster R-CNN varyantını karşılaştırır:
- Temel: Standart Conv2d, toplu iş boyutu 8, LR 0.02
- FDConv: Frekans Dinamik Evrişim, toplu iş boyutu 4, LR 0.005
- WDConv: Wavelet Dinamik Evrişim, toplu iş boyutu 4, LR 0.005

**GPU Runtime:** 40GB VRAM'e sahip A100 GPU

**Ana yollar (Hücre 4'te düzenleyin):**
- `DATA_ROOT`: PASCAL VOC 2007'nin indirilip depolanacağı yer
- `OUTPUT_DIR`: Sonuçların Drive'a kaydedileceği yer

**Veri seti gereksinimleri:**
- PASCAL VOC 2007 (torchvision tarafından otomatik indirilir)
- Gerekli depolama: ~1.5 GB
- Not defteri, indirme ve çıkarma işlemini otomatik olarak gerçekleştirir

**Requirements (otomatik olarak yüklenir):**
- torch, torchvision, torchmetrics, pycocotools, timm, tqdm, matplotlib, pandas, numpy

## Özel Modüller

Her iki not defteri, özel uygulamalara dayanır:

**wdconv.py**: Wavelet Dinamik Evrişim modülü
- 2D FFT'yi 2D Haar Ayrık Wavelet Dönüşümü ile değiştirir
- Alt bant modülasyonu için WaveletBandModulation (WBM) sağlar
- FDConv'dan KernelSpatialModulation (KSM) modüllerini yeniden kullanır

**engine.py**:
- ImageNet64Dataset: Pickle formatı ImageNet-64x64 yüklemesini işler
- Karışık kesinlik (bfloat16) desteği ile eğitim döngüsü
- Kapsamlı metrik izleme (doğruluk, kayıp eğrileri)
- Kontrol noktası kaydetme ve yükleme

**engine.py**:
- Isınma öğrenme oranı planlama ile eğitim
- Gradyan kırpma ve karışık kesinlik (bfloat16)
- Kontrol noktası yönetimi
- torchmetrics.MeanAveragePrecision kullanarak değerlendirme

**models.py**:
- ResNet-50 FPN backbone'u ile Faster R-CNN
- Dinamik evrişim katmanı enjeksiyonu (FDConv veya WDConv)
- Varyant seçimi için yapılandırma sistemi

**metrics.py**:
- mAP değerlendirmesi (mAP, mAP@50, mAP@75)
- Sınıf başına mAP izleme
- FPS kıyaslaması

**voc_dataset.py**:
- torchvision.datasets.VOCDetection etrafında VOCDetectionDataset wrapper
- Veri çoğaltma desteği (sadece yatay çevirme)
- Postprocess işlemleri

## Sonuçlar Yapısı

### fdconv_classification_results/
- `summary_table.csv`: Doğruluk ve hız karşılaştırması
- `benchmark_results.json`: Ayrıntılı eğitim metrikleri
- `variant_times.json`: Varyant başına çalışma süresi ölçümleri
- `runs/`: Ayrı alt dizinler (Temel, FDConv, WDConv) içeren:
  - Checkpoint'ler (best.pth, last.pth)
  - Metrikler (metrics.json)
  - Eğitim kaybı (losses.json)
  - Model bilgisi (model_info.json)
- Görselleştirmeler (PNG dosyaları): Eğitim eğrileri, doğruluk karşılaştırmaları, ağırlık dağılımları

### fdconv_detection_results/
- `summary.csv`: mAP ve FPS karşılaştırması
- `speed_results.json`: Varyant başına zaman ölçümü
- `variant_times.json`: Çalışma süresi ayrıntılandırması
- `runs/`: Checkpoint'ler ve metrikleri içeren ayrı alt dizinler
- Görselleştirmeler (PNG dosyaları): Eğitim eğrileri, tahminler, sınıf başına AP

## Gereksinimler Özeti

Colab GPU ile çalıştırılması tavsiye edilir.  
Tüm paketler, her not defterindeki kurulum hücreleri tarafından yüklenir.  
Sınıflandırma için ImageNet 64x64 verisetinin (pickle formatında) indirilmiş olması gerekmektedir.  
Nesne tespiti için PASCAL VOC 2007 otomatik indirilir.

## Notlar

- Sınıflandırma deneyleri T4 GPU üzerinde ImageNet-64x64 alt kümesini kullanır
- Nesne tespiti deneyleri A100 GPU (40GB VRAM) üzerinde PASCAL VOC 2007'yi kullanır
- Karışık kesinlik (bfloat16) verimlilik için kullanılır
- Kontrol noktaları ve sonuçlar, kalıcılık için Google Drive'a kaydedilir
- Not defterleri, kurulum sırasında resmi FDConv havuzunu klonlar
- WDConv uygulaması, not defterleri içinde kendi kendine yeterlidir
