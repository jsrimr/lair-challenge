from torch import nn, optim
from ttach.base import Merger
from torch.optim.lr_scheduler import _LRScheduler
from datetime import datetime
from typing import Any, List

import numpy as np
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
import ttach as tta
from pytorch_lightning import LightningModule
from sklearn.metrics import f1_score
from torchvision import models

from hyperparams import BATCH_SIZE, ROOT_DIR
from utils import accuracy_function


def accuracy_function(real, pred):
    real = real.cpu()
    pred = torch.argmax(pred, dim=1).cpu()
    score = f1_score(real, pred, average='macro')
    return score


# class CNN_Encoder(nn.Module):
#     def __init__(self, class_n, rate=0.1):
#         super(CNN_Encoder, self).__init__()
#         self.model = models.resnet50(pretrained=True)

#     def forward(self, inputs):
#         output = self.model(inputs)
#         return output


class LSTM_Decoder(nn.Module):
    def __init__(self, max_len, embedding_dim, num_features, class_n, rate):
        super(LSTM_Decoder, self).__init__()
        self.lstm = nn.LSTM(max_len, embedding_dim)
        self.rnn_fc = nn.Linear(num_features*embedding_dim, 1000)
        # resnet out_dim + lstm out_dim
        self.final_layer = nn.Linear(1000 + 1000, class_n)
        self.dropout = nn.Dropout(rate)

    def forward(self, enc_out, dec_inp):
        hidden, _ = self.lstm(dec_inp)
        hidden = hidden.view(hidden.size(0), -1)
        hidden = self.rnn_fc(hidden)
        concat = torch.cat([enc_out, hidden], dim=1)  # enc_out + hidden
        fc_input = concat
        output = self.dropout((self.final_layer(fc_input)))
        return output


# Warmup Learning rate scheduler


class WarmUpLR(_LRScheduler):
    """warmup_training learning rate scheduler
    Args:
        optimizer: optimzier(e.g. SGD)
        total_iters: totoal_iters of warmup phase
    """

    def __init__(self, optimizer, total_iters, last_epoch=-1):

        self.total_iters = total_iters
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """we will use the first m batches, and set the learning
        rate to base_lr * m / total_iters
        """
        return [base_lr * self.last_epoch / (self.total_iters + 1e-8) for base_lr in self.base_lrs]


class CNN2RNNModel(LightningModule):
    def __init__(
        self,
        max_len,
        embedding_dim,
        num_features,
        class_n,
        rate=0.1,
        learning_rate=5e-4,
        cnn_model_name=None,
        tta=False,
        img_size=None
    ):
        super().__init__()

        # self.cnn = CNN_Encoder(class_n)
        self.cnn = timm.create_model(cnn_model_name, pretrained=True,
                                     drop_path_rate=0.2)
        self.rnn = LSTM_Decoder(max_len, embedding_dim,
                                num_features, class_n, rate)

        self.criterion = nn.CrossEntropyLoss()

        self.learning_rate = learning_rate

        self.tta = tta
        self.cnn_model_name = cnn_model_name
        self.img_size = img_size
        # self.max_score = 0

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)

        # optimizer = optim.Lamb(self.parameters(), lr=self.learning_rate, weight_decay=5e-4)
        # warmup_scheduler = WarmUpLR(self.optimizer, iter_per_epoch * args.warm_epoch)

        return optimizer

    def forward(self, img, seq):
        cnn_output = self.cnn(img)
        # return cnn_output
        output = self.rnn(cnn_output, seq)

        return output

    def training_step(self, batch, batch_idx):
        img = batch['img']
        csv_feature = batch['csv_feature']
        label = batch['label']

        output = self(img, csv_feature)
        loss = self.criterion(output, label)
        score = accuracy_function(label, output)

        self.log(
            'train_loss', loss, prog_bar=True, logger=True
        )
        self.log(
            'train_score', score, prog_bar=True, logger=True
        )

        return {'loss': loss, 'train_score': score}

    def on_validation_epoch_start(self):
        self.val_scores = []

    # def on_validation_epoch_end(self):
    #     self.max_score = max(self.max_score, np.mean(self.val_scores))
    #     self.log(
    #         'max_score', self.max_score, prog_bar=True, logger=True
    #     )

    def validation_step(self, batch, batch_idx):
        img = batch['img']
        csv_feature = batch['csv_feature']
        label = batch['label']

        output = self(img, csv_feature)
        loss = self.criterion(output, label)
        score = accuracy_function(label, output)

        self.log(
            'val_loss', loss, prog_bar=True, logger=True
        )
        self.log(
            'score', score, prog_bar=True, logger=True
        )
        # self.val_scores.append(score)

        return {'val_loss': loss, 'score': score}

    def on_predict_epoch_start(self) -> None:
        self.transforms = tta.aliases.d4_transform()
        # self.tta_model = tta.ClassificationTTAWrapper(self, tta.aliases.d4_transform())

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        img = batch['img']
        seq = batch['csv_feature']

        if self.tta:
            merger = Merger(type='mean', n=len(self.transforms))
            for transformer in self.transforms:
                augmented_image = transformer.augment_image(img)
                augmented_output = self(augmented_image, seq)
                merger.append(augmented_output)
            result = merger.result
            output = torch.argmax(result, dim=1)
        else:
            logits = self(img, seq)
            output = torch.argmax(logits, dim=1)

        return {"output": output, "logits": logits}

    def on_predict_epoch_end(self, results: List[Any]) -> None:
        import os
        import pickle
        
        save_path = 'results'
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        save_path = os.path.join(
            save_path, f'{self.cnn_model_name}_{self.img_size}_{datetime.now().strftime("%m%d%H%M")}.pkl')
        with open(save_path, "wb") as f:
            pickle.dump(results, f)
