from ultralytics.data.utils import check_det_dataset
data=check_det_dataset("/home/mq/data_disk2T/Thang/bak/src/data1/data.yaml")
print(data)
self.train_loader = self.get_dataloader(self.trainset, batch_size=batch_size, rank=LOCAL_RANK, mode="train")