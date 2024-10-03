import os
import torch
import json
import albumentations as A
from albumentations.pytorch import ToTensorV2

def save_config(config, filename='config.json'):
    with open(filename, 'w') as f:
        json.dump(config, f, indent=4)


def load_config(filename='config.json'):
    with open(filename, 'r') as f:
        return json.load(f)


CONFIG = {
    "root_dir": os.path.join('C:/Users/lsj/Desktop/YB/TBP/isic-2024-challenge'), ##### CHANGE THIS PATH TO YOUR PATH! ##### FOR KAGGLE SUBMISSION, IT WILL BE "/kaggle/input/isic-2024-challenge"
    "train_dir": os.path.join('C:/Users/lsj/Desktop/YB/TBP/isic-2024-challenge/train-image/image'), ##### CHANGE THIS PATH TO YOUR PATH! ##### FOR KAGGLE SUBMISSION, IT WILL BE f'{ROOT_DIR}/train-image/image'
    "p_upsample_ratio": 5,
    "p:n_ratio": 5,
    "seed": 42,
    "n_epochs": 100,
    "img_size": 384,
    "model_name": "efficientnet_v2_m",
    # "checkpoint_path": None, # Store our pretrained model
    "train_batch_size": 32,
    "valid_batch_size": 64,
    "learning_rate": 5e-3,
    "scheduler": 'CosineAnnealingLR',
    "min_lr": 1e-6,
    "T_max": 100, # set same with epochs -> finish one cycle when train is finished
    "weight_decay": 1e-6,
    #"fold": 0,
    #"n_fold": 5,
    "n_accumulate": 1,
    "n_workers": 8,
    "device": torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
}


TRANSFORM = {
    "train":A.Compose([
            A.Resize(CONFIG['img_size'], CONFIG['img_size']),
            A.RandomRotate90(p=0.5),
            A.Flip(p=0.5),
            A.Downscale(p=0.25),
            A.ShiftScaleRotate(shift_limit=0.1, 
                            scale_limit=0.15, 
                            rotate_limit=60, 
                            p=0.5),
            A.HueSaturationValue(
                    hue_shift_limit=0.2, 
                    sat_shift_limit=0.2, 
                    val_shift_limit=0.2, 
                    p=0.5
                ),
            A.RandomBrightnessContrast(
                    brightness_limit=(-0.1,0.1), 
                    contrast_limit=(-0.1, 0.1), 
                    p=0.5
                ),
            A.Normalize(
                    mean=[0.485, 0.456, 0.406], 
                    std=[0.229, 0.224, 0.225], 
                    max_pixel_value=255.0, 
                    p=1.0
                ),
            ToTensorV2()], p=1.),
    
    "valid":A.Compose([
            A.Resize(CONFIG['img_size'], CONFIG['img_size']),
            A.Normalize(
                    mean=[0.485, 0.456, 0.406], 
                    std=[0.229, 0.224, 0.225], 
                    max_pixel_value=255.0, 
                    p=1.0
                ),
            ToTensorV2()], p=1.)
}