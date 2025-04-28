import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
import numpy as np

# Veri Seti Hazırlama (CIFAR-10) [cite: 13]
print("CIFAR-10 veri seti hazırlanıyor...")
# CIFAR-10 için genellikle kullanılan dönüşümler ve normalizasyon
transform = transforms.Compose([
    transforms.Resize(224), # VGG gibi modeller genellikle 224x224 girdi bekler
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # ImageNet istatistikleri
])

train_data = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
test_data = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_data, batch_size=32, shuffle=True) # Batch size küçültüldü
test_loader = DataLoader(test_data, batch_size=100, shuffle=False)
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
print("CIFAR-10 veri seti hazır.")

# Model Tanımı (Pretrained VGG16) [cite: 12, 13]
print("Önceden eğitilmiş VGG16 modeli yükleniyor...")
model3 = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1) # weights parametresi güncel kullanım [cite: 13]

# VGG16'nın son katmanını CIFAR-10'un 10 sınıfına uyacak şekilde değiştir
num_ftrs = model3.classifier[6].in_features
model3.classifier[6] = nn.Linear(num_ftrs, len(classes))
print("Modelin son katmanı CIFAR-10 için uyarlandı.")

# GPU Kullanımı (varsa)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Kullanılacak cihaz: {device}")
model3 = model3.to(device)

# Eğitim ve Test Fonksiyonları (GPU desteği eklendi)
def train_model(model, train_loader, criterion, optimizer, device, num_epochs=2): # Epoch sayısı demo için azaltıldı
    print("Eğitim başlıyor...")
    model.train()
    loss_history = []
    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if (i+1) % 100 == 0:
                 print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}')
        epoch_loss = running_loss / len(train_loader)
        loss_history.append(epoch_loss)
        print(f'Epoch {epoch+1} tamamlandı, Ortalama Kayıp: {epoch_loss:.4f}')
    print("Eğitim tamamlandı.")
    return loss_history

def test_model(model, test_loader, device):
    print("Test başlıyor...")
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_preds.extend(predicted.cpu().numpy()) # Tahminleri CPU'ya al
            all_labels.extend(labels.cpu().numpy()) # Etiketleri CPU'ya al
    accuracy = 100 * correct / total
    print(f'Test Doğruluğu: {accuracy:.2f}%')
    print("Test tamamlandı.")
    return accuracy, all_labels, all_preds

def plot_loss(loss_history, filename='model3_loss_curve.png'):
    plt.figure()
    plt.plot(loss_history)
    plt.title('Model Kayıp Grafiği (VGG16 - Pretrained)')
    plt.ylabel('Kayıp (Loss)')
    plt.xlabel('Epoch')
    plt.savefig(filename)
    print(f"Kayıp grafiği '{filename}' olarak kaydedildi.")
    # plt.show()

def plot_confusion_matrix(true_labels, pred_labels, classes, filename='model3_confusion_matrix.png'):
    cm = confusion_matrix(true_labels, pred_labels)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.xlabel('Tahmin Edilen Etiket')
    plt.ylabel('Gerçek Etiket')
    plt.title('Karmaşıklık Matrisi (VGG16 - Pretrained)')
    plt.savefig(filename)
    print(f"Karmaşıklık matrisi '{filename}' olarak kaydedildi.")
    # plt.show()


# Ana Çalıştırma Bloğu
if __name__ == '__main__':
    # Hiperparametreler
    learning_rate = 0.0001 # Pretrained model için daha düşük learning rate
    num_epochs = 2 # Demo için kısa tutuldu
    batch_size = 32

    # Kayıp fonksiyonu ve optimizör
    criterion = nn.CrossEntropyLoss()
    # Sadece son katmanın parametrelerini eğitmek için (fine-tuning)
    # optimizer = optim.Adam(model3.classifier[6].parameters(), lr=learning_rate)
    # Veya tüm modeli eğitmek için:
    optimizer = optim.Adam(model3.parameters(), lr=learning_rate)

    print("\n--- Model 3: Pretrained VGG16 (CIFAR-10) ---")
    loss_hist = train_model(model3, train_loader, criterion, optimizer, device, num_epochs=num_epochs)
    accuracy, true_labels, pred_labels = test_model(model3, test_loader, device)

    # Sonuçları Görselleştir
    plot_loss(loss_hist, filename='model3_loss_curve.png')
    plot_confusion_matrix(true_labels, pred_labels, classes=classes, filename='model3_confusion_matrix.png')

    # Modeli kaydetme (isteğe bağlı)
    # torch.save(model3.state_dict(), 'model3_vgg16_pretrained.pth')
    # print("Model 3 'model3_vgg16_pretrained.pth' olarak kaydedildi.")