import os
import cv2
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from torch.nn.utils.rnn import pad_sequence

from config import CONFIG, TRANSFORM
from preprocess import num_cols, cat_cols, final_csv_path, final_img_path

class MelanomaDataset(Dataset):
    def __init__(self, csv_file=final_csv_path, img_dir=final_img_path, num_cols=num_cols, cat_cols=cat_cols, transform=TRANSFORM['train']):
        if isinstance(csv_file, str):
            self.dataframe = pd.read_csv(csv_file)
        elif isinstance(csv_file, pd.DataFrame):
            self.dataframe = csv_file
        else:
            raise ValueError("csv_file must be a file path or a pandas DataFrame")
        self.img_dir = img_dir
        self.num_cols = num_cols
        self.cat_cols = cat_cols
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Extract the file path and load the image
        img_path = self.dataframe.iloc[idx]['file_path']
        img_name = os.path.join(self.img_dir, os.path.basename(img_path))
        if not os.path.isfile(img_name):
            print(f"Image not found: {img_name}")
            return None

        image = cv2.imread(img_name)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if image is None:
            print(f"Failed to load image: {img_name}")
            return None

        # Extract numerical and categorical metadata
        num_metadata = self.dataframe.iloc[idx][self.num_cols].values.astype('float32')
        cat_metadata = self.dataframe.iloc[idx][self.cat_cols].values.astype('float32')
        metadata = np.concatenate((num_metadata, cat_metadata), axis=0)
        metadata = torch.tensor(metadata, dtype=torch.float32)

        # Extract the target label
        label = self.dataframe.iloc[idx]['target']
        label = torch.tensor(label, dtype=torch.float32)

        if isinstance(image, torch.Tensor):
            image = image.numpy()
            
        # Apply transformations to the image if provided
        if self.transform:
            image = self.transform(image=image)['image']

        image = torch.tensor(image, dtype=torch.float32).clone().detach()
            
        return {'image': image, 'metadata': metadata, 'label': label}


def custom_collate_fn(batch):
    images = [item['image'] for item in batch]
    metadata = [item['metadata'] for item in batch]
    labels = [item['label'] for item in batch]

    images = pad_sequence(images, batch_first=True)
    metadata = pad_sequence(metadata, batch_first=True)
    labels = torch.tensor(labels, dtype=torch.float32)

    return {'image': images, 'metadata': metadata, 'label': labels}
    
def prepare_loaders(df_path):
    df = pd.read_csv(df_path)
    train_df, val_df = train_test_split(df, test_size=0.2, stratify=df['target'])

    train_dataset = MelanomaDataset(csv_file=train_df, img_dir=final_img_path, num_cols=num_cols, cat_cols=cat_cols, transform=TRANSFORM['train'])
    val_dataset = MelanomaDataset(csv_file=val_df, img_dir=final_img_path, num_cols=num_cols, cat_cols=cat_cols, transform=TRANSFORM['valid'])

    train_loader = DataLoader(train_dataset, batch_size=CONFIG['train_batch_size'], num_workers=CONFIG['n_workers'], shuffle=True, pin_memory=True, collate_fn=custom_collate_fn)
    valid_loader = DataLoader(val_dataset, batch_size=CONFIG['valid_batch_size'], num_workers=CONFIG['n_workers'], shuffle=False, pin_memory=True, collate_fn=custom_collate_fn)
    
    return train_loader, valid_loader


# Below is for img & tabular, but no upsample case
'''
class MelanomaDataset(Dataset):
    def __init__(self, csv, img_dir, transform=None):
        self.csv = csv
        self.img_dir = img_dir
        self.transform = transform

        if csv is not None and mode != "test":
            self.patient_0 = csv.query(f"target == 0").reset_index(drop=True)
            self.patient_1 = csv.query(f"target == 1").reset_index(drop=True)
        else:
            self.hdf5 = hdf5
            self.patient_ids = list(self.hdf5.keys())
        self.mode = mode
        self.use_meta = meta_features is not None
        self.meta_features = meta_features
        self.transform = transform

    def __len__(self):
        return self.patient_0.shape[0] if self.csv is None else len(self.patient_ids)

    def __getitem__(self, index):
        if self.mode != "test":
            if random.random() > 0.5:
                row = self.patient_1.iloc[index % len(self.patient_1)]
            else:
                row = self.patient_0.iloc[index % len(self.patient_0)]
            image = cv2.imread(row.image_path)
        
        else:
            if self.use_meta:
                row = self.csv.iloc[index]
            image_data = self.hdf5[self.patient_ids[index]][()]
            image = cv2.imdecode(np.frombuffer(image_data, np.uint8), cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform is not None:
            res = self.transform(image=image)
            image = res['image'].astype(np.float32)
        else:
            image = image.astype(np.float32)

        image = image.transpose(2, 0, 1)

        if self.use_meta:
            meta_data = row[self.meta_features].to_numpy().astype(np.float32)
            data = (torch.tensor(image).float(), torch.tensor(meta_data).float())
        else:
            data = torch.tensor(image).float()

        if self.mode == 'test':
            return data
        else:
            return data, torch.tensor(row.target).long()
'''


# Below is for img only, no upsample case
'''
class ISICDataset_for_Train(Dataset):
    def __init__(self, df, transforms=None):
        self.df_positive = df[df["target"] == 1].reset_index()
        self.df_negative = df[df["target"] == 0].reset_index()
        self.file_names_positive = self.df_positive['file_path'].values
        self.file_names_negative = self.df_negative['file_path'].values
        self.targets_positive = self.df_positive['target'].values
        self.targets_negative = self.df_negative['target'].values
        self.transforms = transforms
        
    def __len__(self):
        return len(self.df_positive) * 2
    
    def __getitem__(self, index):
        if random.random() >= 0.5:
            df = self.df_positive
            file_names = self.file_names_positive
            targets = self.targets_positive
        else:
            df = self.df_negative
            file_names = self.file_names_negative
            targets = self.targets_negative
        index = index % df.shape[0]
        
        img_path = file_names[index]
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        target = targets[index]
        
        if self.transforms:
            img = self.transforms(image=img)["image"]
            
        return {
            'image': img,
            'target': target
        }
    
class ISICDataset(Dataset):
    def __init__(self, df, transforms=None):
        self.df = df
        self.file_names = df['file_path'].values
        self.targets = df['target'].values
        self.transforms = transforms
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        img_path = self.file_names[index]
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        target = self.targets[index]
        
        if self.transforms:
            img = self.transforms(image=img)["image"]
            
        return {
            'image': img,
            'target': target
        }
'''
