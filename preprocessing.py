import cv2
import albumentations as A

def get_training_augmentation():
    train_transform = [

        A.HorizontalFlip(p=0.5)

    ]
    return A.Compose(train_transform)


def get_validation_augmentation():
    test_transform = [
        
        A.HorizontalFlip(p=0.5),
        
    ]
    return A.Compose(test_transform)


def get_preprocessing(preprocessing_fn):
    transforms = [A.Resize(height=224, width=224, always_apply=True, p=1, interpolation=cv2.INTER_CUBIC), 
                  A.Lambda(image=preprocessing_fn, mask=preprocessing_fn)]

    return A.Compose(transforms)