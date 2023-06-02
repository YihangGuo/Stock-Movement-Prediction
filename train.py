from __future__ import division, print_function

import argparse
import glob
import os
import pickle
import random
import time
from collections import defaultdict
from os.path import abspath, dirname
from posixpath import join
import logging
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from models import GAT
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, matthews_corrcoef,
                             precision_score, recall_score)
from torch.autograd import Variable
from utils import accuracy, load_data  # load_data: load relaton data
from datetime import datetime
from settings import *
from stock_dataset import StockDataset, PriceTextDataset, StockDataset2, StockDataset4
from torch.utils.data import Dataset, DataLoader
from data_preprocess_train_test_split import get_valid_company_list



def parse_options(parser):
    # Training settings
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training.')
    parser.add_argument('--device', type=int, default=4)
    parser.add_argument('--fastmode', action='store_true', default=False, help='Validate during training pass.')
    parser.add_argument('--sparse', action='store_true', default=False, help='GAT with sparse version or not.')
    parser.add_argument('--seed', type=int, default=14, help='Random seed.')
    # parser.add_argument('--epochs', type=int, default=5000, help='Number of epochs to train.')
    parser.add_argument('--epochs', type=int, default=150, help='Number of epochs to train.')
    parser.add_argument('--lr', type=float, default=5e-4, help='Initial learning rate.')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')
    parser.add_argument('--hidden', type=int, default=64, help='Number of hidden units.')
    parser.add_argument('--nb_heads', type=int, default=8, help='Number of head attentions.')
    parser.add_argument('--dropout', type=float, default=0.38, help='Dropout rate (1 - keep probability).')
    parser.add_argument('--alpha', type=float, default=0.2, help='Alpha for the leaky_relu.')
    parser.add_argument('--patience', type=int, default=100, help='Patience')
    parser.add_argument('--time', type=str, default=datetime.now().strftime("%Y%m%d%H%M%S"))
    parser.add_argument('--eval', action='store_false')

    return parser.parse_args()

def fix_seed(seed=37):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.


class Trainer(object):
    def __init__(self, training=True):
        parser = argparse.ArgumentParser()
        self.args = parse_options(parser)
        self.args.cuda = True
        self.device = self.args.device

        # Set the random seed manually for reproducibility.
        self.seed = self.args.seed
        fix_seed(self.seed)

        # Load data for GAT
        self.adj = load_data()
        self.stock_num = self.adj.size(0)

        # model
        self.model = None
        if training:
            self.model = GAT(nfeat=64, 
                        nhid=self.args.hidden, 
                        nclass=2, 
                        dropout=self.args.dropout, 
                        nheads=self.args.nb_heads, 
                        alpha=self.args.alpha,
                        stock_num=self.stock_num)

        # train and test paths
        self.train_price_path = join(PROJ_DIR, "datasets", "train_price")
        self.train_label_path = join(PROJ_DIR, "datasets", "train_label")
        self.train_text_path = join(PROJ_DIR, "datasets", "train_text")
        self.test_price_path = join(PROJ_DIR, "datasets", "test_price")
        self.test_label_path = join(PROJ_DIR, "datasets", "test_label")
        self.test_text_path = join(PROJ_DIR, "datasets", "test_text")
        self.num_samples = len(os.listdir(self.train_price_path))
        self.cross_entropy = nn.CrossEntropyLoss()
        self.lr = self.args.lr
        self.batch_size = 1
        self.optimizer = optim.Adam(self.model.parameters(), 
                lr=self.args.lr, 
                weight_decay=self.args.weight_decay)

        company_list = get_valid_company_list(join(PROJ_DIR, "valid_company.csv"))
        
        # Create StockDataset instances
        self.train_dataset = StockDataset4(company_list, stage='train')
        # self.train_dataset = StockDataset(company_list, TRAIN_SIZE, train_rate=0.7, is_train=True)
        # self.train_dataset = PriceTextDataset(company_list, train=True)
        # self.train_dataset = StockDataset2(company_list, TRAIN_SIZE, train_rate=0.7, is_train=True)
        self.train_dataloader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

        self.test_dataset = StockDataset4(company_list, stage='test')
        # self.test_dataset = StockDataset(company_list, TEST_SIZE, train_rate=0.7, is_train=False)
        # self.test_dataset = PriceTextDataset(company_list, train=False)
        # self.test_dataset = StockDataset2(company_list, TRAIN_SIZE, train_rate=0.7, is_train=False)
        self.test_dataloader = DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=True)

        self.val_dataset = StockDataset4(company_list, stage='val')
        self.val_dataloader = DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=True)

        self.config_path = '{}_{}_{}_{}_{}'.format(
            self.stock_num,
            self.args.nb_heads,
            self.args.lr,
            self.args.weight_decay,
            self.args.time
            )


    def save_model(self, model, epoch):
        os.makedirs(join(PROJ_DIR, 'checkpoints', self.config_path), exist_ok=True)
        torch.save(model, join(PROJ_DIR, 'checkpoints', self.config_path, "model"
                                + "_epoch_" + str(epoch)
                                + '.ckpt'))


    def train(self, start=0):
        fix_seed(self.seed)
        cuda_device = self.device
        if torch.cuda.is_available():
            torch.cuda.device(cuda_device)
        if self.args.cuda:
            self.model.cuda()
            self.adj = self.adj.cuda()

        Path(join(PROJ_DIR, 'log')).mkdir(parents=True, exist_ok=True)
        
        logging.basicConfig(filename=join(PROJ_DIR, 'log', '%s.log'%self.config_path), level=logging.INFO)
        val_loss_means = []
        val_accuracy_means = []
        val_iops = []
        val_mats = []
        epochs = []

        train_loss_mean = []

        print("===============train dataset length======================={}".format(len(self.train_dataset)))
        print("===============test dataset length======================={}".format(len(self.test_dataset)))

        for epoch in range(start, self.args.epochs):
            self.model.train()
            self.optimizer.zero_grad()
            train_loss = []
            train_acc = []
            
            for i, (train_price, train_text, train_label) in enumerate(self.train_dataloader):
                train_text = train_text.squeeze(0).float().cuda()
                train_price = train_price.squeeze(0).float().cuda()
                train_label = train_label.squeeze(0).cuda()
                
                output = self.model(train_text, train_price, self.adj)
                loss_train = self.cross_entropy(output, train_label)
                acc_train = accuracy(output, train_label)
                
                print(f"Epoch: {epoch}, Iteration: {i}, Training loss = {loss_train.item()}, Accuracy = {acc_train.item()}")
                logging.info("Epoch: %s, Training loss = %s, Accuracy = %s", epoch, loss_train.item(), acc_train.item())
                
                loss_train.backward()
                self.optimizer.step()

                train_loss.append(loss_train.item())
                train_acc.append(acc_train.item())
            
            self.save_model(self.model, epoch)
            val_loss_mean, val_accuracy_mean, val_iop, val_mat = self.validate(epoch)

            epochs.append(epoch)
            val_loss_means.append(val_loss_mean)
            val_accuracy_means.append(val_accuracy_mean)
            val_iops.append(val_iop)
            val_mats.append(val_mat)

            train_loss_mean.append(np.array(train_loss).mean())

        print("Optimization Finished!")
        logging.info("Optimization Finished!")

        plt.plot(epochs, train_loss_mean)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss')

        Path(join(PROJ_DIR, 'plots')).mkdir(parents=True, exist_ok=True)
        # Save the plot as a figure
        plt.savefig(join(PROJ_DIR, 'plots', 'training_{}_{}_{}_{}_{}_{}'.format(
            epoch,
            self.stock_num,
            self.args.nb_heads,
            self.args.lr,
            self.args.weight_decay,
            self.args.time
            ) + ".png"))


        fig, axs = plt.subplots(2, 2, figsize=(14, 10))
        # Plot your data on each subplot
        axs[0, 0].plot(epochs, val_loss_means)
        axs[0, 0].set_title("loss vs epoch")
        axs[0, 1].plot(epochs, val_accuracy_means)
        axs[0, 1].set_title("accuracy vs epoch")
        axs[1, 0].plot(epochs, val_iops)
        axs[1, 0].set_title("iops vs epoch")
        axs[1, 1].plot(epochs, val_mats)
        axs[1, 1].set_title("mats vs epoch (test)")

        # Adjust spacing between subplots
        fig.subplots_adjust(wspace=0.3, hspace=0.3)
        # Add a title
        fig.suptitle("epoch = %s, stock_num = %s, lr = %s" % (epoch, self.stock_num, self.args.lr), fontsize=15)
        # Save the figure to the specified filepath
        Path(join(PROJ_DIR, 'plots')).mkdir(parents=True, exist_ok=True)
        fig.savefig(join(PROJ_DIR, 'plots', '{}_{}_{}_{}_{}_{}'.format(
            epoch,
            self.stock_num,
            self.args.nb_heads,
            self.args.lr,
            self.args.weight_decay,
            self.args.time
            ) + ".png"))
        
        eval_loss_mean, eval_accuracy_mean, eval_iop, eval_mat = self.evaluate(epoch)



    def evaluate(self, epoch):
        logging.info("Evaluate at epoch {}: ...".format(epoch))
        print("Evaluate at epoch {}: ...".format(epoch))
        self.model.eval()
        test_acc = []
        test_loss = []
        li_pred = []
        li_true = []

        with torch.no_grad():
            for (test_price, test_text, test_label) in self.test_dataloader:
                # test_text, test_price, test_label = batch
                test_text = test_text.squeeze(0).float().cuda()
                test_price = test_price.squeeze(0).float().cuda()
                test_label = test_label.squeeze(0).cuda()
                # test_text = test_text.to(device=self.device, dtype=torch.float32)
                # test_price = test_price.to(device=self.device, dtype=torch.float32)
                # test_label = test_label.to(device=self.device, dtype=torch.long)
                output = self.model(test_text, test_price, self.adj)
                loss_test = self.cross_entropy(output, test_label)
                acc_test = accuracy(output, test_label)
                a = output.argmax(1).cpu().numpy()
                b = test_label.cpu().numpy() 
                li_pred.append(a)
                li_true.append(b)
                test_loss.append(loss_test.item())
                test_acc.append(acc_test.item())
        
        iop = f1_score(np.array(li_true).reshape((-1,)),np.array(li_pred).reshape((-1,)), average='micro')
        mat = matthews_corrcoef(np.array(li_true).reshape((-1,)),np.array(li_pred).reshape((-1,)))
        
        loss_mean = np.array(test_loss).mean()
        accuracy_mean = np.array(test_acc).mean()

        print("Test set results:",
            "loss= {:.4f}".format(np.array(test_loss).mean()),
            "accuracy= {:.4f}".format(np.array(test_acc).mean()),
            "F1 score={:.4f}".format(iop),
            "MCC = {:.4f}".format(mat))
        logging.info("Test set results:")
        logging.info("loss= {:.4f}".format(np.array(test_loss).mean()))
        logging.info("accuracy= {:.4f}".format(np.array(test_acc).mean()))
        logging.info("F1 score={:.4f}".format(iop))
        logging.info("MCC = {:.4f}".format(mat))
        
        return loss_mean, accuracy_mean, iop, mat
    

    def validate(self, epoch):
        logging.info("Validation at epoch {}: ...".format(epoch))
        print("Validation at epoch {}: ...".format(epoch))
        self.model.eval()
        val_acc = []
        val_loss = []
        li_pred = []
        li_true = []

        with torch.no_grad():
            for (val_price, val_text, val_label) in self.val_dataloader:
                # test_text, test_price, test_label = batch
                val_text = val_text.squeeze(0).float().cuda()
                val_price = val_price.squeeze(0).float().cuda()
                val_label = val_label.squeeze(0).cuda()
                # test_text = test_text.to(device=self.device, dtype=torch.float32)
                # test_price = test_price.to(device=self.device, dtype=torch.float32)
                # test_label = test_label.to(device=self.device, dtype=torch.long)
                output = self.model(val_text, val_price, self.adj)
                loss_val = self.cross_entropy(output, val_label)
                acc_val = accuracy(output, val_label)
                a = output.argmax(1).cpu().numpy()
                b = val_label.cpu().numpy() 
                li_pred.append(a)
                li_true.append(b)
                val_loss.append(loss_val.item())
                val_acc.append(acc_val.item())
        
        iop = f1_score(np.array(li_true).reshape((-1,)),np.array(li_pred).reshape((-1,)), average='micro')
        mat = matthews_corrcoef(np.array(li_true).reshape((-1,)),np.array(li_pred).reshape((-1,)))
        
        loss_mean = np.array(val_loss).mean()
        accuracy_mean = np.array(val_acc).mean()

        print("Validation set results:",
            "loss= {:.4f}".format(np.array(val_loss).mean()),
            "accuracy= {:.4f}".format(np.array(val_acc).mean()),
            "F1 score={:.4f}".format(iop),
            "MCC = {:.4f}".format(mat))
        logging.info("Test set results:")
        logging.info("loss= {:.4f}".format(np.array(val_loss).mean()))
        logging.info("accuracy= {:.4f}".format(np.array(val_acc).mean()))
        logging.info("F1 score={:.4f}".format(iop))
        logging.info("MCC = {:.4f}".format(mat))
        
        return loss_mean, accuracy_mean, iop, mat
    