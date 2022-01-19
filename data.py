import json
import os
import pickle
from glob import glob

import cv2
import numpy as np
import pandas as pd
import torch
from pytorch_lightning import LightningDataModule, LightningModule, Trainer
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

with open('csv_feature_dict.pkl', 'rb') as pkl:
    csv_feature_dict = pickle.load(pkl)

crop = {'1': '딸기', '2': '토마토', '3': '파프리카', '4': '오이', '5': '고추', '6': '시설포도'}
disease = {'1': {'a1': '딸기잿빛곰팡이병', 'a2': '딸기흰가루병', 'b1': '냉해피해', 'b6': '다량원소결핍 (N)', 'b7': '다량원소결핍 (P)', 'b8': '다량원소결핍 (K)'},
           '2': {'a5': '토마토흰가루병', 'a6': '토마토잿빛곰팡이병', 'b2': '열과', 'b3': '칼슘결핍', 'b6': '다량원소결핍 (N)', 'b7': '다량원소결핍 (P)', 'b8': '다량원소결핍 (K)'},
           '3': {'a9': '파프리카흰가루병', 'a10': '파프리카잘록병', 'b3': '칼슘결핍', 'b6': '다량원소결핍 (N)', 'b7': '다량원소결핍 (P)', 'b8': '다량원소결핍 (K)'},
           '4': {'a3': '오이노균병', 'a4': '오이흰가루병', 'b1': '냉해피해', 'b6': '다량원소결핍 (N)', 'b7': '다량원소결핍 (P)', 'b8': '다량원소결핍 (K)'},
           '5': {'a7': '고추탄저병', 'a8': '고추흰가루병', 'b3': '칼슘결핍', 'b6': '다량원소결핍 (N)', 'b7': '다량원소결핍 (P)', 'b8': '다량원소결핍 (K)'},
           '6': {'a11': '시설포도탄저병', 'a12': '시설포도노균병', 'b4': '일소피해', 'b5': '축과병'}}
risk = {'1': '초기', '2': '중기', '3': '말기'}

label_description = {}
for key, value in disease.items():
    label_description[f'{key}_00_0'] = f'{crop[key]}_정상'
    for disease_code in value:
        for risk_code in risk:
            label = f'{key}_{disease_code}_{risk_code}'
            label_description[label] = f'{crop[key]}_{disease[key][disease_code]}_{risk[risk_code]}'


label_encoder = {key: idx for idx, key in enumerate(label_description)}
label_decoder = {val: key for key, val in label_encoder.items()}


torch.multiprocessing.set_sharing_strategy('file_system')


class CustomDataset(Dataset):
    def __init__(
        self, 
        files, 
        csv_feature_dict, 
        label_encoder,
        labels=None,
        transforms=None,
        mode='train',
    ):
        self.mode = mode
        self.files = files
        
        self.csv_feature_dict = csv_feature_dict
        
        if files is not None:
            self.csv_feature_check = [0]*len(self.files)
            self.csv_features = [None]*len(self.files)
            
        self.max_len = 24 * 6
        
        self.label_encoder = label_encoder
        
        self.transforms = transforms

    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, i):
        file = self.files[i]
        file_name = file.split(os.sep)[-1]
        
        # csv
        if self.csv_feature_check[i] == 0:
            csv_path = f'{file}/{file_name}.csv'
            df = pd.read_csv(csv_path)[self.csv_feature_dict.keys()]
            df = df.replace('-', 0)
            # MinMax scaling
            for col in df.columns:
                df[col] = df[col].astype(float) - self.csv_feature_dict[col][0]
                df[col] = df[col] / (self.csv_feature_dict[col][1]-self.csv_feature_dict[col][0])
            # zero padding
            pad = np.zeros((self.max_len, len(df.columns)))
            length = min(self.max_len, len(df))
            pad[-length:] = df.to_numpy()[-length:]
            # transpose to sequential data
            csv_feature = pad.T
            self.csv_features[i] = csv_feature
            self.csv_feature_check[i] = 1
        else:
            csv_feature = self.csv_features[i]
        
        # image
        image_path = f'{file}/{file_name}.jpg'
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        if self.transforms is not None:
            img = self.transforms(image=img)['image']
        
        if self.mode == 'train':
            json_path = f'{file}/{file_name}.json'
            with open(json_path, 'r') as f:
                json_file = json.load(f)
            
            crop = json_file['annotations']['crop']
            disease = json_file['annotations']['disease']
            risk = json_file['annotations']['risk']
            label = f'{crop}_{disease}_{risk}'
            
            return {
                'img': img,
                'csv_feature': torch.tensor(csv_feature, dtype=torch.float32),
                'label': torch.tensor(self.label_encoder[label], dtype=torch.long)
            }
        else:
            return {
                'img': img,
                'csv_feature': torch.tensor(csv_feature, dtype=torch.float32)
            }
    
    
class CustomDataModule(LightningDataModule):
    def __init__(
        self,
        train=None,
        val=None,
        test=None,
        csv_feature_dict=None,
        label_encoder=None,
        train_transforms=None,
        val_transforms=None,
        predict_transforms=None,
        num_workers=32,
        batch_size=8,
    ):
        super().__init__()
        self.train = train
        self.val = val
        self.test = test
        self.csv_feature_dict = csv_feature_dict
        self.label_encoder = label_encoder
        assert self.csv_feature_dict is not None
        assert self.label_encoder is not None
        self.train_transforms = train_transforms
        self.val_transforms = val_transforms
        self.predict_transforms = predict_transforms
        self.num_workers = num_workers
        self.batch_size = batch_size

    def setup(self, stage=None):
        self.train_dataset = CustomDataset(
            self.train, 
            self.csv_feature_dict,
            self.label_encoder,
            transforms=self.train_transforms,
        )
        self.valid_dataset = CustomDataset(
            self.val, 
            self.csv_feature_dict,
            self.label_encoder,
            transforms=self.train_transforms,
        )
        self.predict_dataset = CustomDataset(
            self.test, 
            self.csv_feature_dict,
            self.label_encoder,
            transforms=self.predict_transforms,
            mode='test'
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            pin_memory=True,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.valid_dataset,
            batch_size=self.batch_size,
            pin_memory=True,
            shuffle=False,
            num_workers=self.num_workers,
        )
        
    def predict_dataloader(self):
        return DataLoader(
            self.predict_dataset,
            batch_size=self.batch_size,
            pin_memory=True,
            shuffle=False,
            num_workers=self.num_workers,
        )
