#coding:utf-8
from ultralytics import YOLO

# 加载模型
# 选项1: 使用预训练模型进行微调
#model = YOLO("yolov8n.pt")  # 加载预训练模型

# 选项2: 从头开始训练（不使用预训练模型）
model = YOLO("yolov8n.yaml")  # 仅加载模型配置，不加载预训练权重

# Use the model
if __name__ == '__main__':
    # Use the model
    # 从头开始训练时，建议增加训练轮数和学习率
    results = model.train(data='datasets/helmetData/data.yaml', epochs=300, batch=32, imgsz=640, device='cpu', cache=True, workers=32, pretrained=False)  # 训练模型
    # 将模型转为onnx格式
    # success = model.export(format='onnx')



