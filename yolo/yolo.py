from ultralytics import YOLO

model = YOLO("yolov8n.pt")

# Detectar solo personas, mochilas y vehículos
images = [
    "/home/yr/dev/cv/yolov8-tflite-cpp/assets/bus.jpg",
    "/home/yr/dev/cv/yolov8-tflite-cpp/assets/zidane.jpg"
]

model.predict(
    source=images,
    classes=[0,1,2,3,5,7,24,26,28],
    show=True
)


"""
person
bicycle
car
motorcycle
bus
truck
backpack
handbag
suitcase
"""