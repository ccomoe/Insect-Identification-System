from transformers import SwinForImageClassification

model = SwinForImageClassification.from_pretrained("microsoft/swin-base-patch4-window7-224")
model.save_pretrained("../swin_base_model")  # 保存模型到本地
