import torch
import torch.nn as nn
import torchvision
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Veri Seti Hazırlama (CIFAR-10) - Model 3 ile aynı
print("CIFAR-10 veri seti hazırlanıyor...")
transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
train_data = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
test_data = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
# Daha hızlı özellik çıkarımı için batch size artırılabilir, ancak bellek kısıtlarına dikkat!
train_loader = DataLoader(train_data, batch_size=64, shuffle=False) # Shuffle False olmalı
test_loader = DataLoader(test_data, batch_size=64, shuffle=False)
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
print("CIFAR-10 veri seti hazır.")

# Özellik Çıkarıcı Model (VGG16'nın evrişim katmanları) [cite: 16]
print("Özellik çıkarıcı model (VGG16 evrişim katmanları) yükleniyor...")
feature_extractor = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).features
# Sınıflandırıcı katmanını kaldırıyoruz, sadece özellikler lazım.
feature_extractor.eval() # Değerlendirme modu önemli
print("Özellik çıkarıcı model hazır.")

# GPU Kullanımı (varsa)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Kullanılacak cihaz: {device}")
feature_extractor = feature_extractor.to(device)

# Özellik Çıkarma Fonksiyonu
def extract_features(loader, model, device):
    print("Özellik çıkarımı başlıyor...")
    features = []
    labels_list = []
    with torch.no_grad():
        for i, (images, labels) in enumerate(loader):
            images = images.to(device)
            # Özellikleri çıkar
            output = model(images)
            # Özellikleri düzleştir (flatten) veya Global Average Pooling kullan
            # VGG'nin çıktısı (batch, channels, height, width) şeklinde
            # Basitçe düzleştirebiliriz:
            output_flat = torch.flatten(output, start_dim=1)
            features.append(output_flat.cpu().numpy())
            labels_list.append(labels.numpy())
            if (i+1) % 50 == 0:
                print(f'Batch [{i+1}/{len(loader)}] tamamlandı.')
    print("Özellik çıkarımı tamamlandı.")
    features = np.concatenate(features, axis=0)
    labels_list = np.concatenate(labels_list, axis=0)
    return features, labels_list

# Karmaşıklık Matrisi Fonksiyonu (Scikit-learn için)
def plot_confusion_matrix_sklearn(true_labels, pred_labels, classes, filename='model4_confusion_matrix.png'):
    cm = confusion_matrix(true_labels, pred_labels)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.xlabel('Tahmin Edilen Etiket')
    plt.ylabel('Gerçek Etiket')
    plt.title('Karmaşıklık Matrisi (Hibrit Model - SVM)')
    plt.savefig(filename)
    print(f"Karmaşıklık matrisi '{filename}' olarak kaydedildi.")
    # plt.show()

# Ana Çalıştırma Bloğu
if __name__ == '__main__':
    print("\n--- Model 4: Hibrit (CNN Özellik Çıkarımı + SVM) ---")

    # Özellikleri çıkar veya kayıttan yükle
    feature_dir = './features'
    train_features_file = os.path.join(feature_dir, 'train_features.npy')
    train_labels_file = os.path.join(feature_dir, 'train_labels.npy')
    test_features_file = os.path.join(feature_dir, 'test_features.npy')
    test_labels_file = os.path.join(feature_dir, 'test_labels.npy')

    if not os.path.exists(feature_dir):
        os.makedirs(feature_dir)

    if os.path.exists(train_features_file) and os.path.exists(train_labels_file):
        print("Eğitim özellikleri ve etiketleri dosyadan yükleniyor...")
        train_features = np.load(train_features_file)
        train_labels = np.load(train_labels_file)
    else:
        print("Eğitim verisi için özellikler çıkarılıyor...")
        train_features, train_labels = extract_features(train_loader, feature_extractor, device)
        print(f"Eğitim özellikleri boyutu: {train_features.shape}")
        np.save(train_features_file, train_features) # Özellikleri kaydet [cite: 16]
        np.save(train_labels_file, train_labels)     # Etiketleri kaydet [cite: 16]
        print(f"Eğitim özellikleri '{train_features_file}' olarak kaydedildi.")
        print(f"Eğitim etiketleri '{train_labels_file}' olarak kaydedildi.")

    if os.path.exists(test_features_file) and os.path.exists(test_labels_file):
        print("Test özellikleri ve etiketleri dosyadan yükleniyor...")
        test_features = np.load(test_features_file)
        test_labels = np.load(test_labels_file)
    else:
        print("Test verisi için özellikler çıkarılıyor...")
        test_features, test_labels = extract_features(test_loader, feature_extractor, device)
        print(f"Test özellikleri boyutu: {test_features.shape}")
        np.save(test_features_file, test_features) # Özellikleri kaydet [cite: 16]
        np.save(test_labels_file, test_labels)     # Etiketleri kaydet [cite: 16]
        print(f"Test özellikleri '{test_features_file}' olarak kaydedildi.")
        print(f"Test etiketleri '{test_labels_file}' olarak kaydedildi.")

    # SVM Modelini Eğitme [cite: 17]
    print("SVM modeli eğitiliyor...")
    # Parametreler ayarlanabilir (C, kernel, gamma vb.)
    svm_classifier = SVC(kernel='linear', C=1.0, random_state=42) # Örnek parametreler
    # Veri miktarını azaltarak daha hızlı eğitim (opsiyonel, demo için)
    # sample_size = 5000
    # train_features = train_features[:sample_size]
    # train_labels = train_labels[:sample_size]
    svm_classifier.fit(train_features, train_labels)
    print("SVM modeli eğitildi.")

    # SVM Modelini Test Etme
    print("SVM modeli test ediliyor...")
    test_predictions = svm_classifier.predict(test_features)
    accuracy = accuracy_score(test_labels, test_predictions)
    print(f'Hibrit Model (SVM) Test Doğruluğu: {accuracy * 100:.2f}%')

    # Karmaşıklık Matrisini Çizdirme
    plot_confusion_matrix_sklearn(test_labels, test_predictions, classes, filename='model4_confusion_matrix.png')