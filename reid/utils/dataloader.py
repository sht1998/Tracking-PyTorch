from torchvision import datasets, transforms
from utils.utils import RandomErasing

def preprocess(h, w, erase_rate):
    transform_train_list = [
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0),
        transforms.Resize((h, w), interpolation=3),
        transforms.Pad(10),
        transforms.RandomCrop((h, w)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        RandomErasing(probability=erase_rate, mean=[0.0, 0.0, 0.0])
    ]

    transform_val_list = [
        transforms.Resize(size=(h, w),interpolation=3), #Image.BICUBIC
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]
    
    train_transformer = transforms.Compose(transform_train_list)
    val_transformer = transforms.Compose(transform_val_list)

    return train_transformer, val_transformer
