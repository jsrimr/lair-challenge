# %%
from hyperparams import batch_size, learning_rate, embedding_dim, max_len, dropout_rate, epochs, vision_pretrain
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from glob import glob
import torch
from torch import nn
import torchvision

import torchvision.transforms as transforms
import torch.nn.functional as F
from torch.utils.data import Dataset

import pytorch_lightning as pl
from pytorch_lightning import Trainer
from argparse import ArgumentParser

from data import csv_feature_dict, label_encoder, label_decoder, LAIRDataModule
from my_model import LitModel
from hyperparams import batch_size



if __name__ == "__main__":
    pl.seed_everything(1234)

    lair_data = LAIRDataModule(batch_size)
    # model = LitModel(max_len=max_len, embedding_dim=embedding_dim, num_features=len(
    #     csv_feature_dict), class_n=len(label_encoder), rate=dropout_rate)
    ckpt_path = "lightning_logs/version_3/checkpoints/epoch=8-step=170.ckpt"
    model = LitModel.load_from_checkpoint(ckpt_path)

    trainer = Trainer(max_epochs=epochs, gpus=1)

    lair_data.setup(stage='test')  # 아, 이걸 직접 콜 해야하나..
    model.eval()  # 아, 이걸 직접 콜 해야하나2..
    output = trainer.predict(model, datamodule=lair_data)

    # ======
    results = torch.argmax(torch.cat(output, axis=0), dim=1)
    # ======
    # results.extend(torch.tensor(torch.argmax(output, dim=1), dtype=torch.int32).cpu().numpy())
    preds = np.array([label_decoder[int(val)] for val in results])
    

    submission = pd.read_csv('data/sample_submission.csv')
    submission['label'] = preds
    submission.to_csv('baseline_submission.csv', index=False)