import os
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from transformers import SwinForImageClassification

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

# Swin Transformer 模型定义
class SwinTransformerModel(nn.Module):
    def __init__(self, num_classes):
        super(SwinTransformerModel, self).__init__()
        self.swin = SwinForImageClassification.from_pretrained("microsoft/swin-base-patch4-window7-224")
        self.swin.classifier = nn.Linear(self.swin.config.hidden_size, num_classes)

    def forward(self, x):
        return self.swin(pixel_values=x).logits

# 加载并初始化模型和类别标签
def load_model_and_classes(model_path, dataset_path='./dataset'):
    # 加载检查点
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    # 检查是否包含类别标签
    if isinstance(checkpoint, dict) and "classes" in checkpoint:
        classes = checkpoint["classes"]
        print("Loaded classes from checkpoint.")
    else:
        # 如果没有保存类别标签，从指定的数据集路径加载类别
        if os.path.exists(dataset_path):
            classes = sorted(os.listdir(dataset_path))
            print(f"Classes not found in checkpoint. Loaded classes from dataset at '{dataset_path}'.")
        else:
            raise FileNotFoundError(f"Dataset path '{dataset_path}' does not exist. Cannot infer classes.")
    
    # 获取类别数
    num_classes = len(classes)
    
    # 初始化模型
    model = SwinTransformerModel(num_classes=num_classes).to(device)
    
    # 加载模型权重
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        checkpoint = torch.load(model_path, map_location=device, weights_only=True)
        # 适配直接保存为 state_dict 的情况
        model.load_state_dict(checkpoint)
    
    model.eval()  # 切换到评估模式
    
    print("Model and classes loaded successfully.")
    return model, classes

# 单张图片预测函数
def predict_image(image_path, model, transform, classes, device):
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0).to(device)  # 增加 batch 维度

    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
        class_name = classes[predicted.item()]
    return class_name

# 文件夹预测函数
def predict_folder(folder_path, model, transform, classes, device):
    predictions = {}
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        if file_name.startswith('.') or not os.path.isfile(file_path):
            continue
        try:
            class_name = predict_image(file_path, model, transform, classes, device)
            predictions[file_name] = class_name
        except Exception as e:
            print(f"Error predicting {file_name}: {e}")
    return predictions

# 通用预测函数
def predict(input_path, model, transform, classes, device):
    if os.path.isfile(input_path):  # 如果是文件，进行单张图片预测
        class_name = predict_image(input_path, model, transform, classes, device)
        print(f"图片: {input_path}, 预测类别: {class_name}")
    elif os.path.isdir(input_path):  # 如果是文件夹，进行文件夹预测
        predictions = predict_folder(input_path, model, transform, classes, device)
        for file_name, class_name in predictions.items():
            print(f"图片: {file_name}, 预测类别: {class_name}")
    else:
        print(f"无效路径: {input_path}")  # 如果路径无效

# 获取用户输入的路径或文件名
input_path = input(f"请输入目标文件名（带扩展名）或文件夹路径（默认路径：./test_images）：").strip()

# 如果用户没有输入，默认使用 './test_images'
if not input_path:
    input_path = './test_images'

# 如果用户输入的是文件名，则在默认路径 './test_images' 中查找该文件
if os.path.isfile(os.path.join('./test_images', input_path)):
    input_path = os.path.join('./test_images', input_path)

model_path = "./models/swin_insect_classifier.pth"
model, classes = load_model_and_classes(model_path)

# 进行预测
predict(input_path, model, transform, classes, device)
