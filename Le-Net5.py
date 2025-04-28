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

# Veri Seti Hazırlama (model1_lenet.py ile aynı)
print("Veri seti hazırlanıyor...")
transform = transforms.Compose([
    transforms.Pad(2),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
train_data = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_data = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = DataLoader(test_data, batch_size=1000, shuffle=False)
print("Veri seti hazır.")

# Model Tanımı (LeNet-5 + Batch Normalization) [cite: 8, 9, 10]
class LeNet5Improved(nn.Module):
    def __init__(self):
        super(LeNet5Improved, self).__init__()
        # Evrişimli katmanlar (Batch Normalization ile)
        self.conv_layers = nn.Sequential(OrderedDict([
            ('c1', nn.Conv2d(1, 6, kernel_size=(5, 5))),
            ('bn1', nn.BatchNorm2d(6)), # Batch Norm eklendi [cite: 10]
            ('relu1', nn.ReLU()),
            ('s2', nn.MaxPool2d(kernel_size=(2, 2), stride=2)),
            ('c3', nn.Conv2d(6, 16, kernel_size=(5, 5))),
            ('bn2', nn.BatchNorm2d(16)), # Batch Norm eklendi [cite: 10]
            ('relu3', nn.ReLU()),
            ('s4', nn.MaxPool2d(kernel_size=(2, 2), stride=2)),
            ('c5', nn.Conv2d(16, 120, kernel_size=(5, 5))),
            ('bn3', nn.BatchNorm2d(120)), # Batch Norm eklendi [cite: 10]
            ('relu5', nn.ReLU())
        ]))
        # Tam bağlantılı katmanlar (Batch Normalization ile)
        self.fc_layers = nn.Sequential(OrderedDict([
            ('f6', nn.Linear(120, 84)),
            ('bn4', nn.BatchNorm1d(84)), # Batch Norm eklendi [cite: 10]
            ('relu6', nn.ReLU()),
            ('f7', nn.Linear(84, 10)),
            ('sig7', nn.LogSoftmax(dim=-1))
        ]))

    def forward(self, x):
        x = self.conv_layers(x)
        x = torch.flatten(x, 1)
        x = self.fc_layers(x)
        return x

# Eğitim ve Test Fonksiyonları (model1_lenet.py'dan kopyalanabilir veya import edilebilir)
# Kolaylık olması için tekrar tanımlandı:
def train_model(model, train_loader, criterion, optimizer, num_epochs=5):
    print("Eğitim başlıyor...")
    model.train()
    loss_history = []
    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, (images, labels) in enumerate(train_loader):
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

def test_model(model, test_loader):
    print("Test başlıyor...")
    model.eval()
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

def plot_loss(loss_history, filename='model2_loss_curve.png'):
    plt.figure()
    plt.plot(loss_history)
    plt.title('Model Kayıp Grafiği (İyileştirilmiş)')
    plt.ylabel('Kayıp (Loss)')
    plt.xlabel('Epoch')
    plt.savefig(filename)
    print(f"Kayıp grafiği '{filename}' olarak kaydedildi.")
    # plt.show()

def plot_confusion_matrix(true_labels, pred_labels, classes, filename='model2_confusion_matrix.png'):
    cm = confusion_matrix(true_labels, pred_labels)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.xlabel('Tahmin Edilen Etiket')
    plt.ylabel('Gerçek Etiket')
    plt.title('Karmaşıklık Matrisi (İyileştirilmiş)')
    plt.savefig(filename)
    print(f"Karmaşıklık matrisi '{filename}' olarak kaydedildi.")
    # plt.show()

# Ana Çalıştırma Bloğu
if __name__ == '__main__':
    # Hiperparametreler
    learning_rate = 0.001
    num_epochs = 5
    batch_size = 64

    # Modeli, kayıp fonksiyonunu ve optimizörü başlat
    model2 = LeNet5Improved()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model2.parameters(), lr=learning_rate)

    print("\n--- Model 2: LeNet-5 + Batch Normalization ---")
    loss_hist = train_model(model2, train_loader, criterion, optimizer, num_epochs=num_epochs)
    accuracy, true_labels, pred_labels = test_model(model2, test_loader)

    # Sonuçları Görselleştir
    plot_loss(loss_hist, filename='model2_loss_curve.png')
    plot_confusion_matrix(true_labels, pred_labels, classes=[str(i) for i in range(10)], filename='model2_confusion_matrix.png')

    # Modeli kaydetme (isteğe bağlı)
    # torch.save(model2.state_dict(), 'model2_lenet_improved.pth')
    # print("Model 2 'model2_lenet_improved.pth' olarak kaydedildi.")