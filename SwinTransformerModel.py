import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from sklearn.model_selection import train_test_split
from transformers import SwinForImageClassification
from tqdm import tqdm
from PIL import Image

# 设备配置
device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 数据预处理
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# 自定义数据集类
class InsectDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = sorted(os.listdir(root_dir))  # 获取昆虫类别
        self.image_paths = []
        self.labels = []

        for idx, class_name in enumerate(self.classes):
            class_folder = os.path.join(root_dir, class_name)
            if not os.path.isdir(class_folder):  # 忽略非文件夹
                continue
            for image_name in os.listdir(class_folder):
                image_path = os.path.join(class_folder, image_name)
                if image_name.startswith('.'):  # 忽略隐藏文件
                    continue
                self.image_paths.append(image_path)
                self.labels.append(idx)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label

# 加载数据集
root_dir = './dataset'  # 替换为实际路径
dataset = InsectDataset(root_dir=root_dir, transform=transform)
train_data, val_data = train_test_split(dataset, test_size=0.2, stratify=dataset.labels)

train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
val_loader = DataLoader(val_data, batch_size=16, shuffle=False)

# Swin Transformer 模型
class SwinTransformerModel(nn.Module):
    def __init__(self, num_classes):
        super(SwinTransformerModel, self).__init__()
        self.swin = SwinForImageClassification.from_pretrained("microsoft/swin-base-patch4-window7-224")
        self.swin.classifier = nn.Linear(self.swin.config.hidden_size, num_classes)

    def forward(self, x):
        return self.swin(pixel_values=x).logits

# 初始化模型
num_classes = len(dataset.classes)
model = SwinTransformerModel(num_classes=num_classes).to(device)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=1e-4)

# 训练函数
def train(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct_preds = 0
    total_preds = 0

    for images, labels in tqdm(train_loader, desc="Training"):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        _, predicted = torch.max(outputs, 1)
        correct_preds += (predicted == labels).sum().item()
        total_preds += labels.size(0)

    accuracy = correct_preds / total_preds
    avg_loss = running_loss / len(train_loader)
    return avg_loss, accuracy

# 验证函数
def validate(model, val_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct_preds = 0
    total_preds = 0

    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="Validating"):
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            correct_preds += (predicted == labels).sum().item()
            total_preds += labels.size(0)

    accuracy = correct_preds / total_preds
    avg_loss = running_loss / len(val_loader)
    return avg_loss, accuracy

# 训练模型
epochs = 10
for epoch in range(epochs):
    print(f"Epoch {epoch+1}/{epochs}")

    train_loss, train_acc = train(model, train_loader, criterion, optimizer, device)
    val_loss, val_acc = validate(model, val_loader, criterion, device)

    print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.4f}")
    print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.4f}")

# 保存模型
torch.save(model.state_dict(), "./model/swin_insect_classifier.pth")
print("Model saved as swin_insect_classifier.pth")
