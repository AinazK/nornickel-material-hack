import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2

def get_train_transforms(image_size: int = 256) -> A.Compose:
    return A.Compose([
        A.Resize(image_size, image_size),  # Быстрый ресайз
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.Affine(
            translate_percent={"x": (-0.05, 0.05), "y": (-0.05, 0.05)},
            scale=(0.9, 1.1),
            rotate=0,
            p=0.3,
            border_mode=cv2.BORDER_REFLECT
        ),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ], is_check_shapes=False)

def get_val_transforms(image_size: int = 256) -> A.Compose:
    return A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ], is_check_shapes=False)