from ultralytics import YOLO

# Load a model
model = YOLO("/home/mq/data_disk2T/Thang/runs/detect/train6/weights/best.pt")

# Chạy dự đoán
results = model.predict(source='/home/mq/data_disk2T/Thang/bak/box/home/mq/data_disk2T/Toan/box/images/0a6ab326b646febbd63e87d75e72f529.jpg', save=False, show=True)
# Xử lý kết quả
print(len(results))
print("hehe")
for result in results:
    boxes = result.boxes.xyxy  # Tọa độ bounding box (x1, y1, x2, y2)
    confidences = result.boxes.conf  # Điểm tin cậy (objectness score)
    class_ids = result.boxes.cls  # Class ID (ID của nhãn lớp)
    print(cls_score)
    print(boxes.size(-2))
    print("Bounding Boxes:", boxes)
    print("Confidences:", confidences)
    print("Class IDs:", class_ids)