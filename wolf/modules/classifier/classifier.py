"""
Covidnet Model
"""

import torch

import matplotlib.pyplot as plt
import numpy as np
import itertools

import numpy as np
import torch.nn as nn

import torch.nn.functional as F
from torchvision import models

from torch import optim
from torch.optim.lr_scheduler import ExponentialLR
import pytorch_lightning as pl

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class PEXP(nn.Module):
    def __init__(self, n_input, n_out):
        super(PEXP, self).__init__()

        '''
        • First-stage Projection: 1×1 convolutions for projecting input features to a lower dimension,
        • Expansion: 1×1 convolutions for expanding features
            to a higher dimension that is different than that of the
            input features,
        • Depth-wise Representation: efficient 3×3 depthwise convolutions for learning spatial characteristics to
            minimize computational complexity while preserving
            representational capacity,
        • Second-stage Projection: 1×1 convolutions for projecting features back to a lower dimension, and
        • Extension: 1×1 convolutions that finally extend channel dimensionality to a higher dimension to produce
             the final features.
             
        '''

        self.network = nn.Sequential(nn.Conv2d(in_channels=n_input, out_channels=n_input // 2, kernel_size=1),
                                     nn.Conv2d(in_channels=n_input // 2, out_channels=int(3 * n_input / 4),
                                               kernel_size=1),
                                     nn.Conv2d(in_channels=int(3 * n_input / 4), out_channels=int(3 * n_input / 4),
                                               kernel_size=3, groups=int(3 * n_input / 4), padding=1),
                                     nn.Conv2d(in_channels=int(3 * n_input / 4), out_channels=n_input // 2,
                                               kernel_size=1),
                                     nn.Conv2d(in_channels=n_input // 2, out_channels=n_out, kernel_size=1))

    def forward(self, x):
        return self.network(x)

class CovidNet(pl.LightningModule):
    # !: when loading model it did not automatically set img_size to (64, 64) and used default size
    def __init__(self, model='large', n_classes=2, img_size=(64, 64), embedding_dim=32, 
            num_channels=1, lr=1e-5, unit_embed=False): # default n_classes=3
        super(CovidNet, self).__init__()
        filters = {
            'pexp1_1': [64, 256],
            'pexp1_2': [256, 256],
            'pexp1_3': [256, 256],
            'pexp2_1': [256, 512],
            'pexp2_2': [512, 512],
            'pexp2_3': [512, 512],
            'pexp2_4': [512, 512],
            'pexp3_1': [512, 1024],
            'pexp3_2': [1024, 1024],
            'pexp3_3': [1024, 1024],
            'pexp3_4': [1024, 1024],
            'pexp3_5': [1024, 1024],
            'pexp3_6': [1024, 1024],
            'pexp4_1': [1024, 2048],
            'pexp4_2': [2048, 2048],
            'pexp4_3': [2048, 2048],
        }
        self.save_hyperparameters()

        self.criterion = nn.CrossEntropyLoss(reduction='mean')

        self.train_acc = pl.metrics.Accuracy()
        self.train_f1 = pl.metrics.F1(num_classes=2, average='macro')

        self.val_acc = pl.metrics.Accuracy(compute_on_step=False)
        self.val_f1 = pl.metrics.F1(num_classes=2, average='macro', compute_on_step=False)
        self.val_precision = pl.metrics.Precision(num_classes=2, average='macro', compute_on_step=False)
        self.val_recall = pl.metrics.Recall(num_classes=2, average='macro', compute_on_step=False)

        self.test_acc = pl.metrics.Accuracy(compute_on_step=False)
        self.test_f1 = pl.metrics.F1(num_classes=2, average='macro', compute_on_step=False)
        self.test_precision = pl.metrics.Precision(num_classes=2, average='macro', compute_on_step=False)
        self.test_recall = pl.metrics.Recall(num_classes=2, average='macro', compute_on_step=False)

        # ! 1 channel
        self.add_module('conv1', nn.Conv2d(in_channels=num_channels, out_channels=64, kernel_size=7, stride=2, padding=3))
        for key in filters:

            if ('pool' in key):
                self.add_module(key, nn.MaxPool2d(filters[key][0], filters[key][1]))
            else:
                self.add_module(key, PEXP(filters[key][0], filters[key][1]))

        if (model == 'large'):

            self.add_module('conv1_1x1', nn.Conv2d(in_channels=64, out_channels=256, kernel_size=1))
            self.add_module('conv2_1x1', nn.Conv2d(in_channels=256, out_channels=512, kernel_size=1))
            self.add_module('conv3_1x1', nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=1))
            self.add_module('conv4_1x1', nn.Conv2d(in_channels=1024, out_channels=2048, kernel_size=1))

            self.__forward__ = self.forward_large_net
        else:
            self.__forward__ = self.forward_small_net
        self.add_module('flatten', Flatten())
        self.add_module('fc1', nn.Linear((self.hparams.img_size[0]**2)*2, 1024))

        self.add_module('fc2', nn.Linear(1024, embedding_dim))
        self.add_module('classifier', nn.Linear(embedding_dim, n_classes)) # self.classifier

    def forward(self, x):
        x = (x - 0.5).div(0.5)
        return self.__forward__(x)

    def predict_from_embedding(self, embed):
        if self.hparams.unit_embed:
            return self.classifier(embed)
        return self.classifier(F.relu(embed))

    def predict(self, x):
        embed = self.forward(x)
        return self.predict_from_embedding(embed)

    def forward_large_net(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        out_conv1_1x1 = self.conv1_1x1(x)

        pepx11 = self.pexp1_1(x)
        pepx12 = self.pexp1_2(pepx11 + out_conv1_1x1)
        pepx13 = self.pexp1_3(pepx12 + pepx11 + out_conv1_1x1)

        out_conv2_1x1 = F.max_pool2d(self.conv2_1x1(pepx12 + pepx11 + pepx13 + out_conv1_1x1), 2)

        pepx21 = self.pexp2_1(
            F.max_pool2d(pepx13, 2) + F.max_pool2d(pepx11, 2) + F.max_pool2d(pepx12, 2) + F.max_pool2d(out_conv1_1x1,
                                                                                                       2))
        pepx22 = self.pexp2_2(pepx21 + out_conv2_1x1)
        pepx23 = self.pexp2_3(pepx22 + pepx21 + out_conv2_1x1)
        pepx24 = self.pexp2_4(pepx23 + pepx21 + pepx22 + out_conv2_1x1)

        out_conv3_1x1 = F.max_pool2d(self.conv3_1x1(pepx22 + pepx21 + pepx23 + pepx24 + out_conv2_1x1), 2)

        pepx31 = self.pexp3_1(
            F.max_pool2d(pepx24, 2) + F.max_pool2d(pepx21, 2) + F.max_pool2d(pepx22, 2) + F.max_pool2d(pepx23,
                                                                                                       2) + F.max_pool2d(
                out_conv2_1x1, 2))
        pepx32 = self.pexp3_2(pepx31 + out_conv3_1x1)
        pepx33 = self.pexp3_3(pepx31 + pepx32 + out_conv3_1x1)
        pepx34 = self.pexp3_4(pepx31 + pepx32 + pepx33 + out_conv3_1x1)
        pepx35 = self.pexp3_5(pepx31 + pepx32 + pepx33 + pepx34 + out_conv3_1x1)
        pepx36 = self.pexp3_6(pepx31 + pepx32 + pepx33 + pepx34 + pepx35 + out_conv3_1x1)

        out_conv4_1x1 = F.max_pool2d(
            self.conv4_1x1(pepx31 + pepx32 + pepx33 + pepx34 + pepx35 + pepx36 + out_conv3_1x1), 2)

        pepx41 = self.pexp4_1(
            F.max_pool2d(pepx31, 2) + F.max_pool2d(pepx32, 2) + F.max_pool2d(pepx32, 2) + F.max_pool2d(pepx34,
                                                                                                       2) + F.max_pool2d(
                pepx35, 2) + F.max_pool2d(pepx36, 2) + F.max_pool2d(out_conv3_1x1, 2))
        pepx42 = self.pexp4_2(pepx41 + out_conv4_1x1)
        pepx43 = self.pexp4_3(pepx41 + pepx42 + out_conv4_1x1)
        flattened = self.flatten(pepx41 + pepx42 + pepx43 + out_conv4_1x1)

        fc1out = F.relu(self.fc1(flattened))
        fc2out = self.fc2(fc1out)
        
        if self.hparams.unit_embed:
            embed = F.normalize(fc2out, dim=1)
        else:
            embed = fc2out

        return embed

    def forward_small_net(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)

        pepx11 = self.pexp1_1(x)
        pepx12 = self.pexp1_2(pepx11)
        pepx13 = self.pexp1_3(pepx12 + pepx11)

        pepx21 = self.pexp2_1(F.max_pool2d(pepx13, 2) + F.max_pool2d(pepx11, 2) + F.max_pool2d(pepx12, 2))
        pepx22 = self.pexp2_2(pepx21)
        pepx23 = self.pexp2_3(pepx22 + pepx21)
        pepx24 = self.pexp2_4(pepx23 + pepx21 + pepx22)

        pepx31 = self.pexp3_1(
            F.max_pool2d(pepx24, 2) + F.max_pool2d(pepx21, 2) + F.max_pool2d(pepx22, 2) + F.max_pool2d(pepx23, 2))
        pepx32 = self.pexp3_2(pepx31)
        pepx33 = self.pexp3_3(pepx31 + pepx32)
        pepx34 = self.pexp3_4(pepx31 + pepx32 + pepx33)
        pepx35 = self.pexp3_5(pepx31 + pepx32 + pepx33 + pepx34)
        pepx36 = self.pexp3_6(pepx31 + pepx32 + pepx33 + pepx34 + pepx35)

        pepx41 = self.pexp4_1(
            F.max_pool2d(pepx31, 2) + F.max_pool2d(pepx32, 2) + F.max_pool2d(pepx32, 2) + F.max_pool2d(pepx34,
                                                                                                       2) + F.max_pool2d(
                pepx35, 2) + F.max_pool2d(pepx36, 2))
        pepx42 = self.pexp4_2(pepx41)
        pepx43 = self.pexp4_3(pepx41 + pepx42)
        flattened = self.flatten(pepx41 + pepx42 + pepx43)

        fc1out = F.relu(self.fc1(flattened))
        #fc2out = F.relu(self.fc2(fc1out))
        fc2out = self.fc2(fc1out)

        if self.hparams.unit_embed:
            embed = F.normalize(fc2out, dim=1)
        else:
            embed = fc2out
        
        return embed

        # ! uncommet bottom 2
        # logits = self.classifier(fc2out)
        # return fc

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=1e-7)
        scheduler = ExponentialLR(optimizer, 0.99)
        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        tensors, targets = batch
        output = self.predict(tensors)
        loss = self.criterion(output, targets)
        self.log('train_loss', loss, prog_bar=True, logger=True)

        self.train_acc(output, targets)
        self.log('train_acc', self.train_acc, on_step=True, on_epoch=False, 
            prog_bar=False, logger=True)

        self.train_f1(output, targets)
        self.log('train_f1', self.train_f1, on_step=True, on_epoch=False, 
            prog_bar=True, logger=True)

        return loss

    def plot_confusion_matrix(self, cm,
                          target_names,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=True):

        accuracy = np.trace(cm) / float(np.sum(cm))
        misclass = 1 - accuracy

        if cmap is None:
            cmap = plt.get_cmap('Blues')

        plt.figure(figsize=(8, 6))
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()

        if target_names is not None:
            tick_marks = np.arange(len(target_names))
            plt.xticks(tick_marks, target_names, rotation=45)
            plt.yticks(tick_marks, target_names)

        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


        thresh = cm.max() / 1.5 if normalize else cm.max() / 2
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            if normalize:
                plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                            horizontalalignment="center",
                            color="white" if cm[i, j] > thresh else "black")
            else:
                plt.text(j, i, "{:,}".format(cm[i, j]),
                            horizontalalignment="center",
                            color="white" if cm[i, j] > thresh else "black")


        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
        self.logger.experiment.log_text('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
        
        # plt.show()
        return plt

    def validation_step(self, batch, batch_idx):
        self.log('curr_epoch', self.current_epoch)
        
        tensors, targets =  batch
        output = self.predict(tensors)
        loss = self.criterion(output, targets)
        self.log('val_loss', loss, prog_bar=True, logger=True, sync_dist=True)

        # """
        # From Dataset:
        # >> (Pdb) self.ct.class_to_idx
        # >> {'CP': 0, 'Normal': 1} (covid positive is 0)
        # """

        # for t, p in zip(targets.cpu().view(-1), preds.cpu().view(-1)):
        #         confusion_matrix[t.long(), p.long()] += 1 # true (row), prediction (col)


        # tn = confusion_matrix[1, 1]  # t = 1 and p = 1 -- true negative
        # tp = confusion_matrix[0, 0]  # t = 0 and p = 0 -- true positive
        # fn = confusion_matrix[0, 1]  # t = 0 and p = 1 -- false negative (BAD, yes covid but we said no)
        # fp = confusion_matrix[1, 0]  # t = 1 and p = 0 -- false positive (false alarm)

        # lg.log_metric('val_tp', tp)
        # lg.log_metric('val_tn', tn)
        # lg.log_metric('val_fn', fn)
        # lg.log_metric('val_fp', fp)

        self.val_acc(output, targets)
        self.log('val_acc', self.val_acc, on_step=False, on_epoch=True, 
                 prog_bar=False, logger=True)

        self.val_f1(output, targets)
        self.log('val_f1', self.val_f1, on_step=False, on_epoch=True,
                 prog_bar=True, logger=True)

        self.val_precision(output, targets)
        self.log('val_precision', self.val_precision, on_step=False, on_epoch=True,
                 prog_bar=False, logger=True)

        self.val_recall(output, targets)
        self.log('val_recall', self.val_recall, on_step=False, on_epoch=True,
                 prog_bar=False, logger=True)

    
    def test_step(self, batch, batch_idx):      
        tensors, targets =  batch
        output = self.predict(tensors)
        loss = self.criterion(output, targets)
        self.log('test_loss', loss, prog_bar=True, logger=True, sync_dist=True)

        self.test_acc(output, targets)
        self.log('test_acc', self.test_acc, on_step=False, on_epoch=True, 
                 prog_bar=False, logger=True)

        self.test_f1(output, targets)
        self.log('test_f1', self.test_f1, on_step=False, on_epoch=True,
                 prog_bar=True, logger=True)

        self.test_precision(output, targets)
        self.log('test_precision', self.test_precision, on_step=False, on_epoch=True,
                 prog_bar=False, logger=True)

        self.test_recall(output, targets)
        self.log('test_recall', self.test_recall, on_step=False, on_epoch=True,
                 prog_bar=False, logger=True)
