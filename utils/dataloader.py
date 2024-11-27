import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np

class ESTM(Dataset):
    def __init__(self, image_dir, confidence_dir, transform, confidence_normalization=False):
        """
        Args:
            Args:
            image_dir (str): Path to the directory containing the images.
            mask_dir (str): Path to the directory containing the labels (masks).
            transform (callable, optional): A function or transformation to apply to both images and labels.
        """
        assert image_dir is not None and confidence_dir is not None, f'image_dir and confidence_dir must be given'
        self.image_dir = image_dir
        self.confidence_dir = confidence_dir
        self.transform = transform
        self.image_filenames = sorted(os.listdir(image_dir))
        self.confidence_filenames = sorted(os.listdir(confidence_dir))
        self.confidence_normalization = confidence_normalization

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        image_name = self.image_filenames[idx]
        image_path = os.path.join(self.image_dir, image_name)
        image = Image.open(image_path).convert("RGB")

        confidence_path = os.path.join(self.confidence_dir, image_name)
        confidence_map = Image.open(confidence_path).convert("L") 
    
        if self.transform:
            image  = self.transform[0](image)
            confidence_map = self.transform[1](confidence_map)
            confidence_map = 255 * ((confidence_map - torch.min(confidence_map)) / (torch.max(confidence_map) - torch.min(confidence_map)))
            if self.confidence_normalization:
                confidence_map = confidence_map / 255.0
                assert torch.min(confidence_map) >= 0.0 and torch.max(confidence_map) <= 1.0, print(f'min:{torch.min(confidence_map)}, max:{torch.max(confidence_map)}')
        
        return image, confidence_map 
    
class Resize:
    def __init__(self, resize_size=532):
        self.resize_size = resize_size
    def __call__(self, image):
        resize_transform = transforms.Resize((self.resize_size, self.resize_size))
        image = resize_transform(image)
        return image

class ToTensor:
    def __call__(self, image):
        image = transforms.ToTensor()(image)
        return image
    
class ToTensor_mask():
    def __init__(self, type='int'):
        self.type = type
    def __call__(self, mask):
        if self.type == 'int':
            mask = torch.from_numpy(np.array(mask, dtype=np.int64))
        elif self.type == 'float':
            mask = torch.from_numpy(np.array(mask, dtype=np.float32))
        else:
            raise ValueError('unknown type')
        return mask

class Normalize:
    def __call__(self, image):
        image = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(image)
        return image

def get_dataloader(data_root, image_size=532, batch_size=8, confidence_normalization=False):
    train_image_dir = os.path.join(data_root, 'images/training')
    val_image_dir = os.path.join(data_root, 'images/validation')
    
    train_confidence_dir = os.path.join(data_root, 'annotations/training')
    val_confidence_dir = os.path.join(data_root, 'annotations/validation')
    
    image_transform = transforms.Compose([Resize(image_size), ToTensor(), Normalize()])
    map_transform = transforms.Compose([Resize(image_size), ToTensor_mask(type='float')])

    train_dataset = ESTM(image_dir=train_image_dir, confidence_dir=train_confidence_dir, \
                            transform=(image_transform, map_transform), \
                            confidence_normalization=confidence_normalization)
    
    val_dataset = ESTM(image_dir=val_image_dir, confidence_dir=val_confidence_dir, \
                        transform=(image_transform, map_transform), \
                        confidence_normalization=confidence_normalization)

    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=4)

    return train_loader, val_loader

if __name__ == "__main__":
    data_root = './dataset'
    train_loader, val_loader = get_dataloader(data_root, 532, batch_size=1, confidence_normalization=False)
    print(len(train_loader), len(val_loader)) 

    train_iter = iter(train_loader)

    image, confidence_map = next(train_iter)

    print(f'image shape: {image.shape}, confidence_map shape: {confidence_map.shape}')
