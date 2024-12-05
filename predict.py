import os
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from transformers import SwinForImageClassification

# 设备配置
device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

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

# 加载类别信息
classes = sorted(os.listdir('./dataset'))  # 替换为实际路径

# 初始化模型
num_classes = len(classes)
model = SwinTransformerModel(num_classes=num_classes).to(device)

# 加载训练好的权重
state_dict = torch.load("./models/swin_insect_classifier.pth", map_location=device, weights_only=True)
model.load_state_dict(state_dict)
model.eval()
print("Model loaded successfully.")

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
    if os.path.isfile(input_path):
        class_name = predict_image(input_path, model, transform, classes, device)
        print(f"Image: {input_path}, Predicted Class: {class_name}")
    elif os.path.isdir(input_path):
        predictions = predict_folder(input_path, model, transform, classes, device)
        for file_name, class_name in predictions.items():
            print(f"Image: {file_name}, Predicted Class: {class_name}")
    else:
        print(f"Invalid path: {input_path}")

# 使用示例
input_path = "./test_images"  # 替换为单张图片路径或文件夹路径
predict(input_path, model, transform, classes, device)
