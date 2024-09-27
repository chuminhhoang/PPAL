import torch
import os

class Class_Quality:
    def __init__(self, num_classes, base_momentum):
        self.num_classes = num_classes
        self.base_momentum = base_momentum
        
        self.class_momentum = torch.ones((num_classes,)) * base_momentum
        self.class_quality = torch.zeros((num_classes,))
        
        self.momentum_file = 'class_momentum.pth'
        self.quality_file = 'class_quality.pth'
        
        self.save_to_file()

    def save_to_file(self):
        torch.save(self.class_momentum, self.momentum_file)
        torch.save(self.class_quality, self.quality_file)

    def load_from_file(self):
        if os.path.exists(self.momentum_file):
            self.class_momentum = torch.load(self.momentum_file)
        else:
            print(f"File {self.momentum_file} không tồn tại!")
            
        if os.path.exists(self.quality_file):
            self.class_quality = torch.load(self.quality_file)
        else:
            print(f"File {self.quality_file} không tồn tại!")

