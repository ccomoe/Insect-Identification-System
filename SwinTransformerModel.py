import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from sklearn.model_selection import train_test_split
from transformers import SwinForImageClassification
from rich.progress import Progress
from PIL import Image

# 设备配置
if torch.backends.mps.is_available():
    device = torch.device("mps")
    device_name = "Apple MPS"
elif torch.cuda.is_available():
    device = torch.device("cuda")
    device_name = torch.cuda.get_device_name(0)  # 获取CUDA设备名称
else:
    device = torch.device("cpu")
    device_name = "CPU"
print(f"Using device: {device} ({device_name})")

# 数据预处理
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# 自定义数据集类
class InsectDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        自定义昆虫数据集类，支持混合层级文件夹结构。

        :param root_dir: 数据集的根目录
        :param transform: 图像变换
        """
        self.root_dir = root_dir
        self.transform = transform
        self.classes = self.get_classes_from_mixed_folders(root_dir)  # 获取类别标签
        self.image_paths = []
        self.labels = []

        # 遍历类别并记录图像路径和对应标签
        for idx, class_name in enumerate(self.classes):
            class_folder = os.path.join(root_dir, class_name.replace('/', os.sep))
            if not os.path.isdir(class_folder):  # 忽略非文件夹
                continue
            for image_name in os.listdir(class_folder):
                if not image_name.lower().endswith(('.png', '.jpg', '.jpeg')) or image_name.startswith('.'):
                    continue
                image_path = os.path.join(class_folder, image_name)
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

    @staticmethod
    def get_classes_from_mixed_folders(dataset_dir):
        """
        获取数据集的类别标签，支持混合层级文件夹结构。
        如果根目录中既有子文件夹又有图像文件，则忽略根目录中的图像文件。
        如果目录为空或无有效类别，返回提示信息并终止程序。
    
        :param dataset_dir: 数据集的根目录
        :return: 类别标签列表
        """
        if not os.path.exists(dataset_dir):
            raise ValueError(f"Dataset directory '{dataset_dir}' does not exist. Please check the path.")
        
        # 初始化类别列表
        classes = []
    
        # 检查根目录中是否既有子文件夹又有图像文件
        root_has_dirs = any(os.path.isdir(os.path.join(dataset_dir, item)) for item in os.listdir(dataset_dir))
        root_has_images = any(
            os.path.isfile(os.path.join(dataset_dir, item)) and item.lower().endswith(('.png', '.jpg', '.jpeg'))
            for item in os.listdir(dataset_dir)
        )
    
        # 如果根目录中有图像文件且有子文件夹，忽略根目录的图像文件
        ignore_root_images = root_has_dirs and root_has_images
    
        # 遍历文件夹结构
        for root, dirs, files in os.walk(dataset_dir):
            # 如果是根目录并且需要忽略图像文件，跳过处理
            if root == dataset_dir and ignore_root_images:
                continue
            # 跳过空文件夹
            if not dirs and not files:
                continue
    
            # 检查当前文件夹是否包含图片文件（根据常见图片后缀判断）
            if any(file.lower().endswith(('.png', '.jpg', '.jpeg')) for file in files):
                # 获取相对路径作为类别标签
                category = os.path.relpath(root, dataset_dir)
                classes.append(category.replace(os.sep, '/'))
    
        # 检查是否成功提取类别
        if not classes:
            raise ValueError(f"No valid classes found in dataset directory '{dataset_dir}'. "
                             "Ensure it contains image files organized in folders.")
        
        return sorted(classes)

# 加载数据集
root_dir = './dataset'  # 替换为实际路径
dataset = InsectDataset(root_dir=root_dir, transform=transform)
train_data, val_data = train_test_split(dataset, test_size=0.2, stratify=dataset.labels)

# 加载类别标签，为后面保存标签使用
classes = dataset.classes

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
    total_batches = len(train_loader)  # 获取总批次数

    with Progress() as progress:
        # 创建任务，设置总进度为 len(train_loader)
        task = progress.add_task("[green]Training [0/{}]".format(total_batches), total=total_batches)

        for batch_idx, (images, labels) in enumerate(train_loader, start=1):
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

            # 更新任务描述，显示当前已训练批次数
            progress.update(task, description=f"[green]Training [{batch_idx}/{total_batches}]")
            # 推进任务进度
            progress.advance(task)

    accuracy = correct_preds / total_preds
    avg_loss = running_loss / len(train_loader)
    return avg_loss, accuracy

# 验证函数
def validate(model, val_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct_preds = 0
    total_preds = 0
    total_batches = len(val_loader)  # 获取总批次数

    with Progress() as progress:
        # 创建任务，设置总进度为 len(Val_loader)
        task = progress.add_task("[cyan]Validating [0/{}]".format(total_batches), total=total_batches)

        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(val_loader, start=1):
                images, labels = images.to(device), labels.to(device)
    
                outputs = model(images)
                loss = criterion(outputs, labels)
    
                running_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                correct_preds += (predicted == labels).sum().item()
                total_preds += labels.size(0)

                # 更新任务描述，显示当前已验证批次数
                progress.update(task, description=f"[cyan]Validating [{batch_idx}/{total_batches}]")
                # 推进任务进度
                progress.advance(task)
            
    accuracy = correct_preds / total_preds
    avg_loss = running_loss / len(val_loader)
    return avg_loss, accuracy

def get_next_model_save_path(base_dir, base_name="swin_insect_classifier"):
    """
    根据已有模型文件自动生成下一个保存路径。
    
    :param base_dir: 模型保存的目录
    :param base_name: 模型的基础名称（默认: swin_insect_classifier）
    :return: 下一个模型保存路径
    """
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)  # 如果目录不存在则创建

    # 获取所有以 base_name 开头且以 .pth 结尾的文件
    existing_files = [
        f for f in os.listdir(base_dir) 
        if f.startswith(base_name) and f.endswith(".pth")
    ]
    
    # 提取文件名中的序号部分
    max_index = -1
    for file in existing_files:
        if file == f"{base_name}.pth":
            # 跳过没有序号的文件
            continue
        try:
            # 提取序号
            index = int(file.replace(base_name, "").replace(".pth", "").strip("_"))
            max_index = max(max_index, index)
        except ValueError:
            continue  # 跳过不符合命名规则的文件

    # 新文件名的序号加 1
    next_index = max_index + 1
    next_file_name = f"{base_name}_{next_index}.pth"

    # 返回完整路径
    return os.path.join(base_dir, next_file_name)

# 训练模型
epochs = 10
for epoch in range(epochs):
    print(f"Epoch {epoch+1}/{epochs}")

    train_loss, train_acc = train(model, train_loader, criterion, optimizer, device)
    val_loss, val_acc = validate(model, val_loader, criterion, device)

    print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.4f}")
    print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.4f}")

# 保存模型和类别标签
base_dir = "./models"
save_path = get_next_model_save_path(base_dir)
torch.save({
    "model_state_dict": model.state_dict(),
    "classes": classes  # 保存类别标签
}, save_path)
print(f"模型已经保存到: {save_path}")
