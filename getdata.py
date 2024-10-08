from ultralytics.data.base import BaseDataset

base=BaseDataset(img_path="/home/mq/data_disk2T/Thang/bak/src/data1/val/labels/0ac4918462e96b54cac4ecaa0f3056fa.txt")
data=base.__getitem__()
print(data)
