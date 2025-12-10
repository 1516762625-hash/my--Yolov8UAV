from ultralytics import YOLO

# 加载你的魔改配置文件
model = YOLO("yolov8-drone.yaml")

# 打印模型信息
model.info()