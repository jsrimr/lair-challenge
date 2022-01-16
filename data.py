from glob import glob
import json
import pickle

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from pytorch_lightning import LightningDataModule, LightningModule, Trainer
from sklearn.model_selection import train_test_split

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


class CustomDataset(Dataset):
    def __init__(self, files, labels=None, mode='train'):
        self.mode = mode
        self.files = files
        self.csv_feature_dict = csv_feature_dict
        self.csv_feature_check = [0]*len(self.files)
        self.csv_features = [None]*len(self.files)
        self.max_len = 24 * 6
        self.label_encoder = label_encoder

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        file = self.files[i]
        file_name = file.split('/')[-1]

        # csv
        if self.csv_feature_check[i] == 0:
            csv_path = f'{file}/{file_name}.csv'
            df = pd.read_csv(csv_path)[self.csv_feature_dict.keys()]
            df = df.replace('-', 0)
            # MinMax scaling
            for col in df.columns:
                df[col] = df[col].astype(float) - self.csv_feature_dict[col][0]
                df[col] = df[col] / (self.csv_feature_dict[col]
                                     [1]-self.csv_feature_dict[col][0])
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
        img = cv2.resize(img, dsize=(256, 256), interpolation=cv2.INTER_AREA)
        img = img.astype(np.float32)/255
        img = np.transpose(img, (2, 0, 1))

        if self.mode == 'train':
            json_path = f'{file}/{file_name}.json'
            with open(json_path, 'r') as f:
                json_file = json.load(f)

            crop = json_file['annotations']['crop']
            disease = json_file['annotations']['disease']
            risk = json_file['annotations']['risk']
            label = f'{crop}_{disease}_{risk}'

            return {
                'img': torch.tensor(img, dtype=torch.float32),
                'csv_feature': torch.tensor(csv_feature, dtype=torch.float32),
                'label': torch.tensor(self.label_encoder[label], dtype=torch.long)
            }
        else:
            return {
                'img': torch.tensor(img, dtype=torch.float32),
                'csv_feature': torch.tensor(csv_feature, dtype=torch.float32)
            }


class LAIRDataModule(LightningDataModule):
    """
    reference : https://colab.research.google.com/drive/16d1uctGaw2y9KhGBlINNTsWpmlXdJwRW#scrollTo=H4Y0pZwX2AsG
    """
    
    def __init__(self, batch_size=32):
        super().__init__()
        self.batch_size = batch_size


    def setup(self, stage=None):
        '''called on each GPU separately - stage defines if we are at fit or test step'''
        # we set up only relevant datasets when stage is specified (automatically set by Pytorch-Lightning)
        train = sorted(glob('data/train/*'))
        test = sorted(glob('data/test/*'))
        labelsss = pd.read_csv('data/train.csv')['label']
        train, val = train_test_split(train, test_size=0.2, stratify=labelsss)

        if stage == 'fit' or stage is None:
            self.train_dataset = CustomDataset(train)
            self.val_dataset = CustomDataset(val)

        if stage == 'test' or stage is None:
            self.test_dataset = CustomDataset(test, mode='test')

    def train_dataloader(self):
        '''returns training dataloader'''
        train_loader = torch.utils.data.DataLoader(
            self.train_dataset, batch_size=self.batch_size, num_workers=8, shuffle=True, persistent_workers=True)
        return train_loader


    def val_dataloader(self):
        '''returns validation dataloader'''
        val_loader = torch.utils.data.DataLoader(
            self.val_dataset, batch_size=self.batch_size, num_workers=8, shuffle=False, persistent_workers=True)
        return val_loader


    def predict_dataloader(self):
        '''returns test dataloader'''
        test_loader = torch.utils.data.DataLoader(
            self.test_dataset, batch_size=self.batch_size, num_workers=8, shuffle=False, persistent_workers=True)
        return test_loader