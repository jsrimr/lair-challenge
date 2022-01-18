from glob import glob
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torchvision import models
from hyperparams import vision_pretrain, learning_rate, batch_size
from utils import accuracy_function

import pytorch_lightning as pl

class CNN_Encoder(nn.Module):
    def __init__(self, class_n, rate=0.1):
        super(CNN_Encoder, self).__init__()
        self.model = models.resnet50(pretrained=vision_pretrain)

    def forward(self, inputs):
        output = self.model(inputs)
        return output


class RNN_Decoder(nn.Module):
    def __init__(self, max_len, embedding_dim, num_features, class_n, rate):
        super(RNN_Decoder, self).__init__()
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


class LitModel(pl.LightningModule):
    def __init__(self, max_len, embedding_dim, num_features, class_n, rate, lr=learning_rate):
        super(LitModel, self).__init__()
        self.save_hyperparameters()

        # architecture
        self.cnn = CNN_Encoder(embedding_dim, rate)
        self.rnn = RNN_Decoder(max_len, embedding_dim,
                               num_features, class_n, rate)

        # etc
        self.learning_rate = lr
        self.batch_size = batch_size


    def forward(self, img, seq):
        cnn_output = self.cnn(img)
        output = self.rnn(cnn_output, seq)

        return output

    def training_step(self, batch, batch_idx):
        # images, labels = batch
        img = batch['img']
        csv_feature = batch['csv_feature']
        label = batch['label']

        # Forward pass
        output = self(img, csv_feature)
        loss = F.cross_entropy(output, label)

        # use key 'log'
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        img = batch['img']
        csv_feature = batch['csv_feature']
        label = batch['label']
        output = self(img, csv_feature)

        score = accuracy_function(label, output)
        loss = F.cross_entropy(output, label)

        # return output
        return {"val_loss": loss, 'score':score}

    def validation_epoch_end(self, outputs):
        # outputs = list of dictionaries
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        # avg_loss = np.average([[x['val_loss'] for x in outputs]])
        # avg_score = sum([x['score'] for x in outputs]) / len(outputs)
        avg_score = np.average([x['score'] for x in outputs])
        # avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        # Forward pass
    
        self.log("val_loss", avg_loss)
        self.log('score', avg_score)
        self.log('lr', self.sch.get_last_lr()[0])
        return {'val_loss': avg_loss, 'score':avg_score}

    def predict_step(self, batch, batch_idx):

        img = batch['img']
        csv_feature = batch['csv_feature']

        # Forward pass
        preds = self(img, csv_feature)

        return preds

    def configure_optimizers(self):
        self.opt = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        self.sch = torch.optim.lr_scheduler.ExponentialLR(self.opt, gamma=0.99)
        return [self.opt], [self.sch]
