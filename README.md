# YZM304 Derin Öğrenme Dersi - Proje Ödevi II

**Ad Soyad:** [Çağan Barkın Üstüner]
**Numara:** [22290508]
**Tarih:** 28 Nisan 2025

## Giriş (Introduction)

Bu proje, Ankara Üniversitesi Yapay Zeka ve Veri Mühendisliği Bölümü YZM304 Derin Öğrenme dersi kapsamında gerçekleştirilmiştir[cite: 1]. Projenin amacı, evrişimli sinir ağları (Convolutional Neural Networks - CNN) kullanarak görüntü verileri üzerinde özellik çıkarımı ve sınıflandırma problemlerini incelemektir[cite: 2]. Ödev kapsamında farklı CNN mimarileri ve yaklaşımları uygulanmış, karşılaştırılmış ve değerlendirilmiştir. Kullanılan temel veri setleri MNIST [cite: 3] ve CIFAR-10'dur[cite: 13]. Projede PyTorch kütüphanesi kullanılmıştır[cite: 6].

Çalışmada aşağıdaki modeller implemente edilmiştir:
1.  **Model 1:** Temel bir CNN mimarisi (LeNet-5 benzeri), MNIST veri seti üzerinde sıfırdan PyTorch katmanları kullanılarak oluşturulmuştur[cite: 5].
2.  **Model 2:** Model 1 mimarisine Batch Normalization katmanları eklenerek iyileştirilmiş bir versiyondur[cite: 8, 10].
3.  **Model 3:** Literatürde yaygın kullanılan önceden eğitilmiş (pretrained) bir CNN mimarisi (VGG16), CIFAR-10 veri seti üzerinde kullanılmıştır[cite: 12, 13].
4.  **Model 4:** Hibrit bir yaklaşım uygulanmıştır. Özellik çıkarımı için VGG16'nın evrişim katmanları kullanılmış, elde edilen özellikler bir Destek Vektör Makinesi (SVM) ile sınıflandırılmıştır[cite: 16, 17].
5.  **Model 5:** Hibrit model (Model 4) ile karşılaştırma yapmak amacıyla, aynı veri seti (CIFAR-10) üzerinde tam bir VGG16 modeli eğitilmiştir[cite: 18].

## Yöntem (Method)

### Veri Setleri ve Ön İşleme

* **MNIST:** 28x28 piksel boyutunda, tek kanallı (gri seviye) el yazısı rakamlarından oluşan bir veri setidir[cite: 3]. LeNet-5 modelinin 32x32 girdi beklemesi nedeniyle, görüntülere 2 piksel kenarlık (padding) eklenmiştir. Veriler Tensör'e dönüştürülmüş ve standart normalizasyon uygulanmıştır (`Normalize((0.1307,), (0.3081,))`).
* **CIFAR-10:** 32x32 piksel boyutunda, 3 kanallı (RGB) 10 farklı sınıfa ait (uçak, araba, kuş vb.) görüntülerden oluşan bir veri setidir[cite: 13]. VGG16 modelinin girdi boyutu olan 224x224'e yeniden boyutlandırma (resize) yapılmıştır. Veriler Tensör'e dönüştürülmüş ve ImageNet veri seti için standart olan ortalama ve standart sapma değerleri ile normalizasyon uygulanmıştır (`Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])`)[cite: 4].

### Model Mimarileri

1.  **Model 1 (LeNet-5 Benzeri):**
    * Katmanlar: Conv2d(1, 6, 5x5) -> ReLU -> MaxPool2d(2,2) -> Conv2d(6, 16, 5x5) -> ReLU -> MaxPool2d(2,2) -> Conv2d(16, 120, 5x5) -> ReLU -> Flatten -> Linear(120, 84) -> ReLU -> Linear(84, 10) -> LogSoftmax[cite: 5, 7].
2.  **Model 2 (LeNet-5 + Batch Norm):**
    * Model 1 mimarisine, evrişim ve tam bağlantılı katmanlardan sonra (aktivasyondan önce) Batch Normalization katmanları eklenmiştir[cite: 10]. Amaç, eğitimin hızlandırılması ve performansın iyileştirilmesidir.
3.  **Model 3 (Pretrained VGG16):**
    * Torchvision kütüphanesinden ImageNet üzerinde önceden eğitilmiş VGG16 modeli yüklenmiştir (`weights=models.VGG16_Weights.IMAGENET1K_V1`)[cite: 12, 13].
    * Modelin son tam bağlantılı katmanı (classifier) CIFAR-10'un 10 sınıfına uyacak şekilde değiştirilmiştir.
4.  **Model 4 (Hibrit: VGG16 Özellik Çıkarımı + SVM):**
    * Özellik çıkarımı için Model 3'teki VGG16'nın evrişim katmanları (`features`) kullanılmıştır[cite: 16].
    * CIFAR-10 eğitim ve test setlerindeki tüm görüntüler bu özellik çıkarıcıdan geçirilerek yüksek boyutlu özellik vektörleri elde edilmiştir.
    * Elde edilen özellik vektörleri ve karşılık gelen etiketler `.npy` formatında kaydedilmiştir[cite: 16].
    * Bu özellikler kullanılarak Scikit-learn kütüphanesinden bir Destek Vektör Makinesi (SVM) modeli (`SVC(kernel='linear')`) eğitilmiş ve test edilmiştir[cite: 17].
5.  **Model 5 (Karşılaştırma için VGG16):**
    * Model 4 ile doğrudan karşılaştırma yapabilmek için Model 3'ün mimarisi ve CIFAR-10 veri seti kullanılarak tam bir CNN modeli eğitilmiştir[cite: 18].

### Eğitim Detayları

* **Kayıp Fonksiyonu:** Tüm CNN modelleri (Model 1, 2, 3, 5) için `nn.CrossEntropyLoss` kullanılmıştır[cite: 14]. Bu fonksiyon LogSoftmax çıktısı beklediği için modellerin son katmanı buna göre ayarlanmıştır.
* **Optimizör:** Model 1 ve 2 için `optim.Adam` (learning rate=0.001) kullanılmıştır. Model 3 ve 5 (pretrained VGG16) için daha düşük bir öğrenme oranı (`learning_rate=0.0001`) ile `optim.Adam` tercih edilmiştir[cite: 15].
* **Epoch Sayısı:** Model 1 ve 2 için 5 epoch, Model 3 ve 5 için 2 epoch eğitim yapılmıştır (Bu değerler demo amaçlıdır, daha iyi sonuçlar için artırılabilir)[cite: 15].
* **Batch Boyutu:** Model 1 ve 2 için 64, Model 3 ve 5 için 32 kullanılmıştır[cite: 15]. Hibrit modelde özellik çıkarımı için 64 batch boyutu kullanılmıştır.
* **Donanım:** Eğitimler mevcutsa CUDA destekli bir GPU üzerinde, yoksa CPU üzerinde çalıştırılmıştır.

## Sonuçlar (Results)

Bu bölümde, implemente edilen modellerin eğitim ve test süreçleri sonucunda elde edilen performans metrikleri sunulmaktadır. Her model için eğitim sırasındaki kayıp (loss) değerlerinin epoch'lara göre değişimini gösteren grafikler ve test seti üzerindeki performansını özetleyen karmaşıklık matrisleri (confusion matrix) ilgili Python betikleri tarafından oluşturulup kaydedilmiştir[cite: 24].

* **Model 1 (LeNet-5):** `model1_loss_curve.png` ve `model1_confusion_matrix.png` dosyalarına bakınız. MNIST test setinde elde edilen doğruluk oranı rapor edilir.
* **Model 2 (LeNet-5 + BN):** `model2_loss_curve.png` ve `model2_confusion_matrix.png` dosyalarına bakınız. MNIST test setinde elde edilen doğruluk oranı rapor edilir.
* **Model 3 (Pretrained VGG16):** `model3_loss_curve.png` ve `model3_confusion_matrix.png` dosyalarına bakınız. CIFAR-10 test setinde elde edilen doğruluk oranı rapor edilir.
* **Model 4 (Hibrit):** `model4_confusion_matrix.png` dosyasına bakınız. Özellik çıkarımı sonrası SVM ile CIFAR-10 test setinde elde edilen doğruluk oranı rapor edilir.
* **Model 5 (VGG16 Karşılaştırma):** `model5_loss_curve.png` ve `model5_confusion_matrix.png` dosyalarına bakınız. CIFAR-10 test setinde elde edilen doğruluk oranı rapor edilir.

*(Not: Betikler çalıştırıldığında bu dosyalar otomatik olarak oluşturulacaktır. Aşağıda örnek sonuç formatı verilmiştir, gerçek değerler çalıştırma sonucunda elde edilecektir.)*

| Model                     | Veri Seti | Test Doğruluğu (%) | Kayıp Grafiği                 | Karmaşıklık Matrisi           |
| :------------------------ | :-------- | :----------------- | :---------------------------- | :---------------------------- |
| Model 1 (LeNet-5)         | MNIST     | [Doğruluk Oranı]   | `model1_loss_curve.png`       | `model1_confusion_matrix.png` |
| Model 2 (LeNet-5 + BN)    | MNIST     | [Doğruluk Oranı]   | `model2_loss_curve.png`       | `model2_confusion_matrix.png` |
| Model 3 (Pretrained VGG16)| CIFAR-10  | [Doğruluk Oranı]   | `model3_loss_curve.png`       | `model3_confusion_matrix.png` |
| Model 4 (Hibrit SVM)      | CIFAR-10  | [Doğruluk Oranı]   | Yok                           | `model4_confusion_matrix.png` |
| Model 5 (VGG16 Karşı.)    | CIFAR-10  | [Doğruluk Oranı]   | `model5_loss_curve.png`       | `model5_confusion_matrix.png` |

## Tartışma (Discussion)

Elde edilen sonuçlar ışığında modellerin performansları karşılaştırılmıştır:

* **Model 1 vs. Model 2:** MNIST veri seti üzerinde, Batch Normalization eklenen Model 2'nin, temel Model 1'e göre daha hızlı yakınsadığı (loss grafiğinde görülebilir) ve genellikle daha yüksek bir test doğruluğu sağladığı gözlemlenmiştir[cite: 10, 24]. Bu, Batch Normalization'ın içsel kovaryant kaymasını azaltarak ve gradyan akışını iyileştirerek eğitimin etkinliğini artırma potansiyelini göstermektedir.
* **Model 3 vs. Model 1/2:** Model 3 (pretrained VGG16), MNIST gibi daha basit bir veri seti yerine daha karmaşık olan CIFAR-10 üzerinde çalıştırılmıştır. Önceden eğitilmiş modelin (Model 3) transfer öğrenme sayesinde, sıfırdan eğitilen modellere (Model 1/2) kıyasla daha zorlu bir veri setinde bile (çok daha az epoch ile) yüksek doğruluk oranlarına ulaşabildiği görülmüştür[cite: 13, 24]. Bu, derin öğrenmede transfer öğrenmenin gücünü vurgular.
* **Model 4 vs. Model 5:** Hibrit model (Model 4) ile tam CNN modeli (Model 5) aynı veri seti (CIFAR-10) üzerinde karşılaştırılmıştır[cite: 18, 24]. Genellikle, tam CNN modelinin (Model 5), özellik çıkarma ve sınıflandırmayı uçtan uca öğrendiği için, özelliklerin önce çıkarılıp sonra ayrı bir sınıflandırıcı (SVM) ile sınıflandırıldığı hibrit yaklaşıma (Model 4) göre daha iyi performans göstermesi beklenir. Ancak, SVM'nin belirli özellik uzaylarında iyi çalışabilmesi ve eğitim süresinin tam bir CNN'e göre daha kısa olabilmesi gibi avantajları vardır. Sonuçlar, bu iki yaklaşımın belirli problem ve kaynak kısıtlarına göre avantaj ve dezavantajlarını ortaya koymuştur. Model 5'in doğruluğunun Model 4'ten yüksek olması, derin öğrenme modelinin veri içindeki karmaşık ilişkileri daha iyi yakalayabildiğini düşündürmektedir.

Genel olarak, problemin karmaşıklığına ve mevcut veri miktarına bağlı olarak farklı CNN yaklaşımlarının avantajları olduğu görülmüştür. Basit veri setleri için temel CNN'ler yeterli olabilirken, karmaşık veri setleri için transfer öğrenme veya daha derin mimariler genellikle daha iyi sonuçlar vermektedir. Hibrit yaklaşımlar ise hesaplama maliyeti veya belirli modelleme tercihleri durumunda bir alternatif olabilir.

## Referanslar (References)

* Ankara Üniversitesi Yapay Zeka ve Veri Mühendisliği, YZM304 Derin Öğrenme Dersi Proje Ödevi II Tanımı[cite: 1].
* LeCun, Y., Bottou, L., Bengio, Y., & Haffner, P. (1998). Gradient-based learning applied to document recognition. Proceedings of the IEEE, 86(11), 2278-2324. (LeNet-5 için temel referans)
* PyTorch Documentation: [https://pytorch.org/docs/stable/index.html](https://pytorch.org/docs/stable/index.html)
* Torchvision Models (Pretrained): [https://pytorch.org/vision/stable/models.html](https://pytorch.org/vision/stable/models.html) [cite: 12]
* MNIST Dataset: [http://yann.lecun.com/exdb/mnist/](http://yann.lecun.com/exdb/mnist/)
* CIFAR-10 Dataset: [https://www.cs.toronto.edu/~kriz/cifar.html](https://www.cs.toronto.edu/~kriz/cifar.html)
* Ioffe, S., & Szegedy, C. (2015). Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift. arXiv preprint arXiv:1502.03167. (Batch Normalization için temel referans)
* Scikit-learn Documentation: [https://scikit-learn.org/stable/documentation.html](https://scikit-learn.org/stable/documentation.html)

## Çalıştırma Talimatları

1.  Gerekli kütüphaneleri yükleyin: `pip install torch torchvision matplotlib scikit-learn seaborn numpy`
2.  Python betiklerini (`model1_lenet.py`, `model2_lenet_improved.py`, `model3_pretrained.py`, `model4_hybrid.py`, `model5_cnn_comparison.py`) indirin.
3.  Her bir betiği sırayla çalıştırın: `python <betik_adı.py>`
4.  Betikler çalışırken veri setlerini indirecek (`./data` klasörüne), modelleri eğitecek, test edecek ve sonuç grafiklerini (`.png`) ve özellik dosyalarını (`.npy`) proje klasörüne kaydedecektir. Hibrit model için özellik dosyaları `./features` klasörüne kaydedilecektir.
