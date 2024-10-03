import os
import json
from PIL import Image

# Đường dẫn đến folder chứa labels và images
labels_folder = '/home/mq/data_disk2T/Thang/bak/src/data1/val/labels'  # Thay đổi đường dẫn này
images_folder = '/home/mq/data_disk2T/Thang/bak/src/data1/val/images'    # Thay đổi đường dẫn này

# Khởi tạo danh sách để lưu thông tin
data = []
batch_size = 8  # Kích thước của mỗi batch
batch_idx = 0   # Chỉ số của batch

# Lấy danh sách các file labels và ảnh
labels_files = sorted(os.listdir(labels_folder))
images_files = sorted(os.listdir(images_folder))

# Kiểm tra xem số lượng file có khớp không
if len(labels_files) != len(images_files):
    raise ValueError("Số lượng file trong thư mục labels không khớp với số lượng file trong thư mục images.")

# Lặp qua từng file label để tạo thông tin cho JSON
for i, label_file in enumerate(labels_files):
    # Đọc file label
    with open(os.path.join(labels_folder, label_file), 'r') as file:
        lines = file.readlines()

    # Lấy thông tin ảnh
    img_name = images_files[i]
    img_path = os.path.join(images_folder, img_name)

    # Mở ảnh để lấy chiều cao và chiều rộng
    with Image.open(img_path) as img:
        width, height = img.size

    # Lặp qua từng dòng trong file label
    for line in lines:
        parts = list(map(float, line.strip().split()))
        if len(parts) >= 5:  # Kiểm tra có đủ thông tin
            label = int(parts[0])      # Nhãn
            x_center = parts[1] * width  # Chuyển đổi về tọa độ ảnh
            y_center = parts[2] * height
            w = parts[3] * width
            h = parts[4] * height

            # Tính toạ độ bbox (x_min, y_min, x_max, y_max)
            x_min = x_center - w / 2
            y_min = y_center - h / 2
            x_max = x_center + w / 2
            y_max = y_center + h / 2
            
            # Tạo một đối tượng để lưu thông tin cho JSON
            bbox_data = {
                "bbox": [x_min, y_min, x_max, y_max],
                "labels": label,
                "batch_idx": batch_idx,
                "image_idx": i % batch_size,  # Vị trí của ảnh trong batch
                "file_name": img_path,
                "width": width,
                "height": height
            }
            data.append(bbox_data)

    # Nếu batch đủ kích thước, reset lại chỉ số batch
    if (i + 1) % batch_size == 0:
        batch_idx += 1

# Sắp xếp dữ liệu theo batch_idx
data = sorted(data, key=lambda x: x['batch_idx'])

# Lưu dữ liệu vào file JSON
with open('output_data.json', 'w') as outfile:
    json.dump(data, outfile, indent=4)

print("File JSON đã được tạo thành công.")
