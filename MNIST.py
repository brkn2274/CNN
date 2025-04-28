import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from collections import OrderedDict
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
import numpy as np

# Veri Seti Hazırlama
print("Veri seti hazırlanıyor...")
transform = transforms.Compose([
    transforms.Pad(2), # LeNet-5 32x32 girdi bekler, MNIST 28x28'dir [cite: 3]
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)) # MNIST için genellikle kullanılan ortalama ve std sapma
])

train_data = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_data = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = DataLoader(test_data, batch_size=1000, shuffle=False)
print(f"Eğitim verisi sayısı: {len(train_data)}")
print(f"Test verisi sayısı: {len(test_data)}")
print("Veri seti hazır.")

# Model Tanımı (LeNet-5 Benzeri) [cite: 5, 6, 7]
class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        # Evrişimli katmanlar
        self.conv_layers = nn.Sequential(OrderedDict([
            ('c1', nn.Conv2d(1, 6, kernel_size=(5, 5))), # 1 girdi kanalı, 6 çıktı kanalı, 5x5 filtre [cite: 7]
            ('relu1', nn.ReLU()), # Aktivasyon [cite: 7]
            ('s2', nn.MaxPool2d(kernel_size=(2, 2), stride=2)), # Havuzlama [cite: 7]
            ('c3', nn.Conv2d(6, 16, kernel_size=(5, 5))),
            ('relu3', nn.ReLU()),
            ('s4', nn.MaxPool2d(kernel_size=(2, 2), stride=2)),
            ('c5', nn.Conv2d(16, 120, kernel_size=(5, 5))), # LeNet-5'te bu tam bağlantılıdır, burada evrişimli bırakıldı.
            ('relu5', nn.ReLU())
        ]))
        # Tam bağlantılı katmanlar (Sınıflandırıcı) [cite: 7]
        self.fc_layers = nn.Sequential(OrderedDict([
            ('f6', nn.Linear(120, 84)),
            ('relu6', nn.ReLU()),
            ('f7', nn.Linear(84, 10)), # MNIST için 10 sınıf çıktısı
            ('sig7', nn.LogSoftmax(dim=-1)) # CrossEntropyLoss ile kullanım için LogSoftmax [cite: 14]
        ]))

    def forward(self, x):
        x = self.conv_layers(x)
        x = torch.flatten(x, 1) # Düzleştirme [cite: 7]
        x = self.fc_layers(x)
        return x

# Eğitim Fonksiyonu
def train_model(model, train_loader, criterion, optimizer, num_epochs=5):
    print("Eğitim başlıyor...")
    model.train() # Eğitim moduna al
    loss_history = []
    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, (images, labels) in enumerate(train_loader):
            # Gradyanları sıfırla
            optimizer.zero_grad()
            # İleri yayılım
            outputs = model(images)
            loss = criterion(outputs, labels)
            # Geri yayılım ve optimizasyon
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

# Test Fonksiyonu
def test_model(model, test_loader):
    print("Test başlıyor...")
    model.eval() # Değerlendirme moduna al
    all_preds = []
    all_labels = []
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_preds.extend(predicted.numpy())
            all_labels.extend(labels.numpy())
    accuracy = 100 * correct / total
    print(f'Test Doğruluğu: {accuracy:.2f}%')
    print("Test tamamlandı.")
    return accuracy, all_labels, all_preds

# Kayıp Grafiği Fonksiyonu
def plot_loss(loss_history):
    plt.figure()
    plt.plot(loss_history)
    plt.title('Model Kayıp Grafiği')
    plt.ylabel('Kayıp (Loss)')
    plt.xlabel('Epoch')
    plt.savefig('model1_loss_curve.png') # Grafiği kaydet
    print("Kayıp grafiği 'model1_loss_curve.png' olarak kaydedildi.")
    # plt.show() # Eğer interaktif ortamda çalıştırılıyorsa göster

# Karmaşıklık Matrisi Fonksiyonu
def plot_confusion_matrix(true_labels, pred_labels, classes, filename='model1_confusion_matrix.png'):
    cm = confusion_matrix(true_labels, pred_labels)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.xlabel('Tahmin Edilen Etiket')
    plt.ylabel('Gerçek Etiket')
    plt.title('Karmaşıklık Matrisi')
    plt.savefig(filename)
    print(f"Karmaşıklık matrisi '{filename}' olarak kaydedildi.")
    # plt.show()

# Ana Çalıştırma Bloğu
if __name__ == '__main__':
    # Hiperparametreler [cite: 15]
    learning_rate = 0.001
    num_epochs = 5 # Demo için kısa tutuldu, daha iyi sonuçlar için artırılabilir
    batch_size = 64

    # Modeli, kayıp fonksiyonunu ve optimizörü başlat
    model1 = LeNet5()
    criterion = nn.CrossEntropyLoss() # [cite: 14]
    optimizer = optim.Adam(model1.parameters(), lr=learning_rate) # [cite: 15]

    print("\n--- Model 1: LeNet-5 Benzeri ---")
    loss_hist = train_model(model1, train_loader, criterion, optimizer, num_epochs=num_epochs)
    accuracy, true_labels, pred_labels = test_model(model1, test_loader)

    # Sonuçları Görselleştir
    plot_loss(loss_hist)
    plot_confusion_matrix(true_labels, pred_labels, classes=[str(i) for i in range(10)])

    # Modeli kaydetme (isteğe bağlı)
    # torch.save(model1.state_dict(), 'model1_lenet.pth')
    # print("Model 1 'model1_lenet.pth' olarak kaydedildi.")