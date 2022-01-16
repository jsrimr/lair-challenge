import os
from argparse import ArgumentParser
from glob import glob

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from pytorch_lightning import Trainer
from torch import nn
from torch.utils.data import Dataset
from torchvision import models
from tqdm import tqdm

from data import LAIRDataModule, csv_feature_dict, label_decoder, label_encoder
from hyperparams import (batch_size, dropout_rate, embedding_dim, epochs,
                         learning_rate, max_len, vision_pretrain)
from my_model import LitModel
from pytorch_lightning.callbacks import ModelCheckpoint


if __name__ == "__main__":
    pl.seed_everything(1234)

    # ------------
    # args
    # ------------
    # parser = ArgumentParser()
    # parser.add_argument('--batch_size', default=32, type=int)
    # parser = pl.Trainer.add_argparse_args(parser)
    # parser = CNN2RNN.add_model_specific_args(parser)
    # args = parser.parse_args()

    lair_data = LAIRDataModule(batch_size)
    model = LitModel(max_len=max_len, embedding_dim=embedding_dim, num_features=len(
        csv_feature_dict), class_n=len(label_encoder), rate=dropout_rate)
    # model.load_state_dict(torch.load(save_path, map_location=device))

    checkpoint_callback = ModelCheckpoint(monitor="score",
                                          save_top_k=3, mode="max")
    trainer = Trainer(max_epochs=epochs, gpus=2, precision=16,
                      default_root_dir=os.getcwd(), callbacks=[checkpoint_callback])

    # trainer.tune(model)
    trainer.fit(model, datamodule=lair_data)
