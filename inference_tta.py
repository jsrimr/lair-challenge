# %%
import copy
from datetime import datetime
import os
from argparse import ArgumentParser
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

from data import (CustomDataModule, csv_feature_dict, label_decoder,
                  label_encoder)
from hyperparams import INF_BATCH_SIZE as BATCH_SIZE
from hyperparams import (CLASS_N, DROPOUT_RATE, EMBEDDING_DIM,
                         EPOCHS, IMAGE_HEIGHT, IMAGE_WIDTH, LEARNING_RATE,
                         MAX_LEN, NUM_FEATURES, NUM_WORKERS, ROOT_DIR, SEED)
from my_model import CNN2RNNModel
from train import split_data
import ttach as tta

def get_predict_transforms(height, width):
    return A.Compose([
        A.Resize(height=height, width=width),
        A.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])


tta_transforms = tta.Compose(
    [
        tta.Rotate90(angles=[0, 90, 180, 270]),
    ]
)


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
    test_data = split_data(mode='test')
    
    predict_transforms = get_predict_transforms(IMAGE_HEIGHT, IMAGE_WIDTH)
    
    # data_module = CustomDataModule(
    #     test=test_data,
    #     csv_feature_dict=csv_feature_dict,
    #     label_encoder=label_encoder,
    #     predict_transforms=predict_transforms,
    #     num_workers=NUM_WORKERS,
    #     batch_size=BATCH_SIZE,
    # )
    from data import CustomDataset, DataLoader
    predict_dataset = CustomDataset(
            test_data, 
            csv_feature_dict,
            label_encoder,
            transforms=predict_transforms,
            mode='test'
        )
    test_data_loader = DataLoader(
            predict_dataset,
            batch_size=BATCH_SIZE,
            pin_memory=True,
            shuffle=False,
            num_workers=NUM_WORKERS,
        )

    model = CNN2RNNModel(
        max_len=24*6, 
        embedding_dim=512, 
        num_features=len(csv_feature_dict), 
        class_n=len(label_encoder),
        tta=True
    )
    ckpt = torch.load(ckpt_path)
    model.load_state_dict(ckpt['state_dict'])

    device = "cuda:0"
    model.to(device)
    model.eval()

    tta_model = tta.ClassificationTTAWrapper(model, tta_transforms)

    test_df = pd.read_csv(f'{ROOT_DIR}/sample_submission.csv')
    submission_df = copy.deepcopy(test_df)

    for i, (batch) in enumerate(tqdm(test_data_loader)):
        images = batch['img'].to(device)
        seq = batch['csv_feature'].to(device)

        outputs = tta_model(images, seq).detach().cpu().numpy().squeeze() # soft
        # outputs = (outputs > 0.5).astype(int) # hard vote
        batch_index = i * BATCH_SIZE
        submission_df.iloc[batch_index:batch_index+BATCH_SIZE, 1:] += outputs
    
    # submission_df.iloc[:,1:] = (submission_df.iloc[:,1:] / len(weights) > 0.35).astype(int)  # ensemble
    
    SAVE_FN = os.path.join('submissions', datetime.now().strftime("%m%d%H%M") + '_submission.csv')

    submission_df.to_csv(
        SAVE_FN,
        index=False
        )


    # get_submission(outputs, submit_save_dir, submit_save_name, label_decoder)


if __name__ == "__main__":
    pl.seed_everything(1234)

    CKPT_PATH = 'weights/ConvNeXt-B-22k/epoch=9-val_score=0.899.ckpt'

    eval(CKPT_PATH, csv_feature_dict, label_encoder, label_decoder)    
