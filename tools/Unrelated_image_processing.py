import os
import torch
from torchvision import models, transforms
from PIL import Image

# 设置数据集文件夹路径
dataset_dir = "/Users/cool/Downloads/animals/cheetah"  # 替换为你的图片根目录路径

# 遍历文件夹获取所有图片路径
dataset = []
for root, _, files in os.walk(dataset_dir):
    for file in files:
        if file.endswith(('.jpg', '.jpeg', '.png', '.bmp')):  # 检查图片格式
            dataset.append(os.path.join(root, file))

print(f"共加载 {len(dataset)} 张图片")

# 加载预训练模型（ResNet50）
model = models.resnet50(pretrained=True)
model.eval()

# 定义图片预处理操作
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# 定义分类函数
def classify_image(image_path):
    try:
        # 打开图片并转换为 RGB 格式
        image = Image.open(image_path).convert("RGB")
        # 对图片进行预处理并转为张量
        input_tensor = preprocess(image).unsqueeze(0)
        # 使用模型预测
        with torch.no_grad():
            output = model(input_tensor)
            confidence, class_id = torch.max(output, 1)
            return confidence.item(), class_id.item()
    except Exception as e:
        print(f"无法处理图片 {image_path}: {e}")
        return None, None

# 设置置信度阈值
confidence_threshold = 0.3

# 筛选无关图片
irrelevant_images = []  # 存储可能无关的图片路径

print("开始筛选无关图片...")
for image_path in dataset:
    confidence, class_id = classify_image(image_path)
    if confidence is not None and confidence < confidence_threshold:
        irrelevant_images.append(image_path)
        print(f"可能的无关图片: {image_path}，置信度: {confidence:.2f}")

print(f"检测到 {len(irrelevant_images)} 张可能的无关图片")

# # 可选：清理无关图片
# delete = input("是否删除检测到的无关图片？(y/n): ").strip().lower()
# if delete == "y":
#     for image_path in irrelevant_images:
#         try:
#             os.remove(image_path)
#             print(f"已删除图片: {image_path}")
#         except Exception as e:
#             print(f"无法删除图片 {image_path}: {e}")

# print("处理完成。")
