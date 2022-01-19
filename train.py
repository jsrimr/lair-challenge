from my_model import CNN2RNNModel
from data import CustomDataModule
import os
import json
from glob import glob

import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models
import pytorch_lightning as pl
from pytorch_lightning import LightningDataModule, LightningModule
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.utilities.seed import seed_everything
import albumentations as A
from albumentations.pytorch import ToTensorV2
from pytorch_lightning.loggers import WandbLogger

import wandb

ROOT_DIR = 'data'


def get_train_transforms(height, width):
    return A.Compose([
        A.Resize(height=height, width=width),
        A.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])


def get_valid_transforms(height, width):
    return A.Compose([
        A.Resize(height=height, width=width),
        A.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ToTensorV2()
    ])


def initialize():
    csv_feature_dict = {
        '내부 온도 1 평균': [3.4, 47.3],
        '내부 온도 1 최고': [3.4, 47.6],
        '내부 온도 1 최저': [3.3, 47.0],
        '내부 습도 1 평균': [23.7, 100.0],
        '내부 습도 1 최고': [25.9, 100.0],
        '내부 습도 1 최저': [0.0, 100.0],
        '내부 이슬점 평균': [0.1, 34.5],
        '내부 이슬점 최고': [0.2, 34.7],
        '내부 이슬점 최저': [0.0, 34.4]
    }

    crop = {'1': '딸기', '2': '토마토', '3': '파프리카',
            '4': '오이', '5': '고추', '6': '시설포도'}
    disease = {
        '1': {
            'a1': '딸기잿빛곰팡이병', 'a2': '딸기흰가루병', 'b1': '냉해피해',
            'b6': '다량원소결핍 (N)', 'b7': '다량원소결핍 (P)', 'b8': '다량원소결핍 (K)'
        },
        '2': {
            'a5': '토마토흰가루병', 'a6': '토마토잿빛곰팡이병', 'b2': '열과', 'b3': '칼슘결핍',
            'b6': '다량원소결핍 (N)', 'b7': '다량원소결핍 (P)', 'b8': '다량원소결핍 (K)'
        },
        '3': {
            'a9': '파프리카흰가루병', 'a10': '파프리카잘록병', 'b3': '칼슘결핍',
            'b6': '다량원소결핍 (N)', 'b7': '다량원소결핍 (P)', 'b8': '다량원소결핍 (K)'
        },
        '4': {
            'a3': '오이노균병', 'a4': '오이흰가루병', 'b1': '냉해피해',
            'b6': '다량원소결핍 (N)', 'b7': '다량원소결핍 (P)', 'b8': '다량원소결핍 (K)'
        },
        '5': {
            'a7': '고추탄저병', 'a8': '고추흰가루병', 'b3': '칼슘결핍',
            'b6': '다량원소결핍 (N)', 'b7': '다량원소결핍 (P)', 'b8': '다량원소결핍 (K)'
        },
        '6': {'a11': '시설포도탄저병', 'a12': '시설포도노균병', 'b4': '일소피해', 'b5': '축과병'}
    }
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

    return csv_feature_dict, label_encoder, label_decoder


def split_data(split_rate=0.2, seed=42, mode='train'):
    """
    Use for model trained image and time series.
    """
    if mode == 'train':
        train = sorted(glob(f'{ROOT_DIR}/train/*'))

        labelsss = pd.read_csv(f'{ROOT_DIR}/train.csv')['label']
        train, val = train_test_split(
            train, test_size=split_rate, random_state=seed, stratify=labelsss)

        return train, val
    elif mode == 'test':
        test = sorted(glob(f'{ROOT_DIR}/test/*'))

        return test


csv_feature_dict, label_encoder, label_decoder = initialize()


def main(config):
    """
    Use for model trained image and time series.
    """
    train_data, val_data = split_data(seed=1234, mode='train')

    train_transforms = get_train_transforms(
        config.IMAGE_HEIGHT, config.IMAGE_WIDTH)
    val_transforms = get_valid_transforms(
        config.IMAGE_HEIGHT, config.IMAGE_WIDTH)

    data_module = CustomDataModule(
        train=train_data,
        val=val_data,
        csv_feature_dict=csv_feature_dict,
        label_encoder=label_encoder,
        train_transforms=train_transforms,
        val_transforms=val_transforms,
        num_workers=config.NUM_WORKERS,
        batch_size=config.BATCH_SIZE,
    )

    model = CNN2RNNModel(
        max_len=config.MAX_LEN,
        embedding_dim=config.EMBEDDING_DIM,
        num_features=config.NUM_FEATURES,
        class_n=config.CLASS_N,
        rate=config.DROPOUT_RATE,
        learning_rate=config.LEARNING_RATE,
    )
    wandb_logger = WandbLogger(project="lair-challenge")
    wandb_logger.watch(model.cnn)
    wandb_logger.watch(model.rnn)
    pl.seed_everything(1234)

    ckpt_path = f'./weights/{config.MODEL_NAME}/'
    if not os.path.exists(ckpt_path):
        os.makedirs(ckpt_path)

    checkpoint = ModelCheckpoint(
        monitor='val_score',
        dirpath=ckpt_path,
        filename='{epoch}-{val_score:.3f}',
        save_top_k=-1,
        mode='max',
        save_weights_only=True,
    )

    trainer = pl.Trainer(
        max_epochs=config.EPOCHS,
        gpus=1,
        precision=16,
        callbacks=[checkpoint],
        log_every_n_steps=5,
        logger=wandb_logger
    )

    trainer.fit(model, data_module)


# Sweep parameters


if __name__ == "__main__":
    hyperparameter_defaults = dict(
        SEED=42,
        IMAGE_WIDTH=256,
        IMAGE_HEIGHT=256,
        BATCH_SIZE=256,
        CLASS_N=len(label_encoder),
        LEARNING_RATE=1e-4,
        EMBEDDING_DIM=512,
        NUM_FEATURES=len(csv_feature_dict),
        MAX_LEN=24*6,
        DROPOUT_RATE=0.1,
        EPOCHS=10,
        NUM_WORKERS=16,
        MODEL_NAME='resnet50'
    )
    wandb.init(config=hyperparameter_defaults)
    config = wandb.config
    main(config)
