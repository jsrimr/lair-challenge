# %%
import os
from argparse import ArgumentParser
from datetime import datetime
from glob import glob

import albumentations as A
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from albumentations.pytorch import ToTensorV2
from pytorch_lightning import Trainer
from torch import nn
from torch.utils.data import Dataset
from tqdm import tqdm

from data import CustomDataModule
from hyperparams import (BATCH_SIZE, CLASS_N, DROPOUT_RATE, EMBEDDING_DIM,
                         EPOCHS, IMAGE_HEIGHT, IMAGE_WIDTH, LEARNING_RATE,
                         MAX_LEN, MODEL_NAME, NUM_FEATURES, NUM_WORKERS, ROOT_DIR, SEED)
from my_model import CNN2RNNModel
from train import split_data
from utils import initialize


def get_predict_transforms(height, width):
    return A.Compose([
        A.Resize(height=height, width=width),
        A.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])


def get_submission(outputs, save_dir, save_filename, label_decoder):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    outputs = [o.detach().cpu().numpy() for batch in outputs
                                        for o in batch]
    preds = np.array([label_decoder[int(val)] for val in outputs])
    
    submission = pd.read_csv(f'{ROOT_DIR}/sample_submission.csv')
    submission['label'] = preds
    
    save_file_path = os.path.join(save_dir, save_filename)
    
    submission.to_csv(save_file_path, index=False)


def eval(
    ckpt_path, 
    csv_feature_dict, 
    label_encoder, 
    label_decoder,
    submit_save_dir='submissions',
    submit_save_name='baseline_submission.csv',
):
    # test_data = split_data(mode='test')
    test_data = sorted(glob(f'{ROOT_DIR}/test/*'))
    
    predict_transforms = get_predict_transforms(IMAGE_HEIGHT, IMAGE_WIDTH)
    
    data_module = CustomDataModule(
        test=test_data,
        csv_feature_dict=csv_feature_dict,
        label_encoder=label_encoder,
        predict_transforms=predict_transforms,
        num_workers=NUM_WORKERS,
        batch_size=BATCH_SIZE,
    )
    
    model = CNN2RNNModel(
        max_len=24*6, 
        embedding_dim=512, 
        num_features=len(csv_feature_dict), 
        class_n=len(label_encoder),
        cnn_model_name=MODEL_NAME,
        img_size=IMAGE_WIDTH
    )

    trainer = pl.Trainer(
        gpus=[args.gpu],
        precision=16,
        # strategy="ddp"
        # fast_dev_run=True,
        # limit_predict_batches=0.01
    )

    ckpt = torch.load(ckpt_path)
    model.load_state_dict(ckpt['state_dict'])

    outputs = trainer.predict(model, data_module)
    outputs = [output['output'] for output in outputs]

    get_submission(outputs, submit_save_dir, submit_save_name, label_decoder)


if __name__ == "__main__":
    pl.seed_everything(1234)

    parser = ArgumentParser()
    parser.add_argument('ckpt_path', type=str, default='weights/convnext_base_384_in22ft1k-288-288/epoch=8-score=1.000.ckpt')
    parser.add_argument('--gpu', type=int, default=1)
    args = parser.parse_args()

    csv_feature_dict, label_encoder, label_decoder = initialize()
    # CKPT_PATH = 'weights/ConvNeXt-B-22k/epoch=14-score=0.996.ckpt'
    # CKPT_PATH = 'weights/ConvNeXt-B-22k/epoch=9-score=1.000.ckpt'
    # CKPT_PATH = 'weights/convnext_xlarge_384_in22ft1k/epoch=8-score=0.998.ckpt'
    #    weights/convnext_xlarge_384_in22ft1k/res512/epoch=9-score=1.000-v2.ckpt
    #    weights/convnext_xlarge_384_in22ft1k/epoch=5-score=1.000-v1.ckpt
    CKPT_PATH = args.ckpt_path

    weight_dir, model_name, info = CKPT_PATH.split('/')
    save_filename = model_name + info.replace(".ckpt", "") + '_submission.csv'
    eval(CKPT_PATH, csv_feature_dict, label_encoder, label_decoder, submit_save_name=save_filename)    
