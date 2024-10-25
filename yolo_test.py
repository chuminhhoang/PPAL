from ultra.ultralytics.models.yolo.model import YOLO
from ultra.ultralytics.utils import ASSETS
from ultra.ultralytics.models.yolo.detect import DetectionPredictor

args = dict(model="/home/mq/data_disk2T/Thang/best.pt", source=ASSETS)

predictor = DetectionPredictor(overrides=args)
print(predictor.inference_feature(['/home/mq/data_disk2T/Thang/bak/src/data/train/images/0a0f62b62a5fdfdee141f232b80dba19.jpg', '/home/mq/data_disk2T/Thang/bak/src/data/train/images/0a0f62b62a5fdfdee141f232b80dba19.jpg']).shape)
# print(predictor.stream_inference(['/home/mq/data_disk2T/Thang/bak/src/data/train/images/0a0f62b62a5fdfdee141f232b80dba19.jpg', '/home/mq/data_disk2T/Thang/bak/src/data/train/images/0a0f62b62a5fdfdee141f232b80dba19.jpg']))
# a = YOLO('/home/mq/data_disk2T/Thang/best.pt')
# a('/home/mq/data_disk2T/Thang/bak/src/data/train/images/0a0f62b62a5fdfdee141f232b80dba19.jpg')