import torch
import numpy as np
import pandas as pd
import os
import pandas as pd
import torch
import pandas_market_calendars as mcal
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from settings import *



class StockDataset5(Dataset):
    def __init__(self, company_list, stage='train'):
        self.company_list = company_list
        # self.data_num = data_num
        # self.train_rate = train_rate
        self.stage = stage
        self.max_len = 0
        for company in self.company_list:
            if self.stage == 'train':
                max_len = len(os.listdir(join(PROJ_DIR, "datasets", "train2", "label", company)))
                self.max_len = max(max_len, self.max_len)
            elif self.stage == 'val':
                max_len = len(os.listdir(join(PROJ_DIR, "datasets", "val2", "label", company)))
                self.max_len = max(max_len, self.max_len)
            else:
                max_len = len(os.listdir(join(PROJ_DIR, "datasets", "test2", "label", company)))
                self.max_len = max(max_len, self.max_len)
        
        if self.stage == 'train':
            print('Build training data...')
        elif self.stage == 'val':
            print('Build validation data...')
        else:
            print('Build test data...')
    
    def __len__(self):
        # return self.data_num
        return self.max_len
    
    def __getitem__(self, index):
        price_list = []
        tweet_list = []
        label_list = []
        for company in self.company_list:
            if self.stage == 'train':
                file_list = os.listdir(join(PROJ_DIR, "datasets", "train2", "label", company))
                total_date_n = len(file_list)
                rand_num = int(index % int(total_date_n))
                target_filename = file_list[rand_num]
                price_path = join(PROJ_DIR, "datasets", "train2", "price", company, target_filename)
                text_path = join(PROJ_DIR, "datasets", "train2", "text", company, target_filename)
                label_path = join(PROJ_DIR, "datasets", "train2", "label", company, target_filename)
            elif self.stage == 'val':
                file_list = os.listdir(join(PROJ_DIR, "datasets", "val2", "label", company))
                total_date_n = len(file_list)
                rand_num = int(index % int(total_date_n))
                target_filename = file_list[rand_num]
                price_path = join(PROJ_DIR, "datasets", "val2", "price", company, target_filename)
                text_path = join(PROJ_DIR, "datasets", "val2", "text", company, target_filename)
                label_path = join(PROJ_DIR, "datasets", "val2", "label", company, target_filename)
            else:
                file_list = os.listdir(join(PROJ_DIR, "datasets", "test2", "label", company))
                total_date_n = len(file_list)
                rand_num = int(index % int(total_date_n))
                target_filename = file_list[rand_num]
                price_path = join(PROJ_DIR, "datasets", "test2", "price", company, target_filename)
                text_path = join(PROJ_DIR, "datasets", "test2", "text", company, target_filename)
                label_path = join(PROJ_DIR, "datasets", "test2", "label", company, target_filename)
                
            # if self.is_train:
            #     file_list = os.listdir(join(PROJ_DIR, "datasets", "raw_data", "label", company))
            #     total_date_n = len(file_list)
            #     # rand_num = random.randint(0,int(total_date_n*self.train_rate)-1)
            #     rand_num = int(index % (int(total_date_n * self.train_rate) - 1))
            #     target_filename = file_list[rand_num]
            # else:
            #     file_list = os.listdir(join(PROJ_DIR, "datasets", "raw_data", "label", company))
            #     total_date_n = len(file_list)
            #     # rand_num = random.randint(int(total_date_n*self.train_rate),total_date_n-1)
            #     rand_num = int(index % (total_date_n - 1 - int(total_date_n * self.train_rate)) + (total_date_n * self.train_rate))
            #     target_filename = file_list[rand_num]
                
            # price = np.load(join(PROJ_DIR, "datasets", "raw_data", "price", company, target_filename))
            # tweet = np.load(join(PROJ_DIR, "datasets", "raw_data", "text", company, target_filename))
            # label = np.load(join(PROJ_DIR, "datasets", "raw_data", "label", company, target_filename))
            price = np.load(price_path)
            tweet = np.load(text_path)
            label = np.load(label_path)

            price_list.append(price)
            tweet_list.append(tweet)
            label_list.append(label)
        
        price_list = np.array(price_list)
        tweet_list = np.array(tweet_list)
        label_list = np.array(label_list)
        
        return price_list, tweet_list, label_list


class StockDataset4(Dataset):
    def __init__(self, company_list, stage='train'):
        self.company_list = company_list
        # self.data_num = data_num
        # self.train_rate = train_rate
        self.stage = stage
        self.max_len = 0
        for company in self.company_list:
            if self.stage == 'train':
                max_len = len(os.listdir(join(PROJ_DIR, "datasets", "train", "label", company)))
                self.max_len = max(max_len, self.max_len)
            elif self.stage == 'val':
                max_len = len(os.listdir(join(PROJ_DIR, "datasets", "val", "label", company)))
                self.max_len = max(max_len, self.max_len)
            else:
                max_len = len(os.listdir(join(PROJ_DIR, "datasets", "test", "label", company)))
                self.max_len = max(max_len, self.max_len)
        
        if self.stage == 'train':
            print('Build training data...')
        elif self.stage == 'val':
            print('Build validation data...')
        else:
            print('Build test data...')
    
    def __len__(self):
        # return self.data_num
        return self.max_len
    
    def __getitem__(self, index):
        price_list = []
        tweet_list = []
        label_list = []
        for company in self.company_list:
            # price_path = ''
            # text_path = ''
            # label_path = ''
            if self.stage == 'train':
                file_list = os.listdir(join(PROJ_DIR, "datasets", "train", "label", company))
                total_date_n = len(file_list)
                rand_num = int(index % int(total_date_n))
                target_filename = file_list[rand_num]
                price_path = join(PROJ_DIR, "datasets", "train", "price", company, target_filename)
                text_path = join(PROJ_DIR, "datasets", "train", "text", company, target_filename)
                label_path = join(PROJ_DIR, "datasets", "train", "label", company, target_filename)
            elif self.stage == 'val':
                file_list = os.listdir(join(PROJ_DIR, "datasets", "val", "label", company))
                total_date_n = len(file_list)
                rand_num = int(index % int(total_date_n))
                target_filename = file_list[rand_num]
                price_path = join(PROJ_DIR, "datasets", "val", "price", company, target_filename)
                text_path = join(PROJ_DIR, "datasets", "val", "text", company, target_filename)
                label_path = join(PROJ_DIR, "datasets", "val", "label", company, target_filename)
            else:
                file_list = os.listdir(join(PROJ_DIR, "datasets", "test", "label", company))
                total_date_n = len(file_list)
                rand_num = int(index % int(total_date_n))
                target_filename = file_list[rand_num]
                price_path = join(PROJ_DIR, "datasets", "test", "price", company, target_filename)
                text_path = join(PROJ_DIR, "datasets", "test", "text", company, target_filename)
                label_path = join(PROJ_DIR, "datasets", "test", "label", company, target_filename)
                
            # if self.is_train:
            #     file_list = os.listdir(join(PROJ_DIR, "datasets", "raw_data", "label", company))
            #     total_date_n = len(file_list)
            #     # rand_num = random.randint(0,int(total_date_n*self.train_rate)-1)
            #     rand_num = int(index % (int(total_date_n * self.train_rate) - 1))
            #     target_filename = file_list[rand_num]
            # else:
            #     file_list = os.listdir(join(PROJ_DIR, "datasets", "raw_data", "label", company))
            #     total_date_n = len(file_list)
            #     # rand_num = random.randint(int(total_date_n*self.train_rate),total_date_n-1)
            #     rand_num = int(index % (total_date_n - 1 - int(total_date_n * self.train_rate)) + (total_date_n * self.train_rate))
            #     target_filename = file_list[rand_num]
                
            # price = np.load(join(PROJ_DIR, "datasets", "raw_data", "price", company, target_filename))
            # tweet = np.load(join(PROJ_DIR, "datasets", "raw_data", "text", company, target_filename))
            # label = np.load(join(PROJ_DIR, "datasets", "raw_data", "label", company, target_filename))
            price = np.load(price_path)
            tweet = np.load(text_path)
            label = np.load(label_path)

            price_list.append(price)
            tweet_list.append(tweet)
            label_list.append(label)
        
        price_list = np.array(price_list)
        tweet_list = np.array(tweet_list)
        label_list = np.array(label_list)
        
        return price_list, tweet_list, label_list


class StockDataset3(Dataset):
    def __init__(self, company_list, is_train=True):
        self.company_list = company_list
        # self.data_num = data_num
        # self.train_rate = train_rate
        self.is_train = is_train
        self.max_len = 0
        for company in self.company_list:
            max_len = len(os.listdir(join(PROJ_DIR, "datasets", "raw_data", "label", company)))
            self.max_len = max(max_len, self.max_len)

        
        if self.is_train:
            print('Build training data...')
        else:
            print('Build test data...')
    
    def __len__(self):
        # return self.data_num
        return self.max_len
    
    def __getitem__(self, index):
        price_list = []
        tweet_list = []
        label_list = []
        for company in self.company_list:
            if self.is_train:
                file_list = os.listdir(join(PROJ_DIR, "datasets", "train", "label", company))
                total_date_n = len(file_list)
                rand_num = int(index % int(total_date_n))
                target_filename = file_list[rand_num]
            else: 
                file_list = os.listdir(join(PROJ_DIR, "datasets", "test", "label", company))
                total_date_n = len(file_list)
                rand_num = int(index % int(total_date_n))
                target_filename = file_list[rand_num]
                
            # if self.is_train:
            #     file_list = os.listdir(join(PROJ_DIR, "datasets", "raw_data", "label", company))
            #     total_date_n = len(file_list)
            #     # rand_num = random.randint(0,int(total_date_n*self.train_rate)-1)
            #     rand_num = int(index % (int(total_date_n * self.train_rate) - 1))
            #     target_filename = file_list[rand_num]
            # else:
            #     file_list = os.listdir(join(PROJ_DIR, "datasets", "raw_data", "label", company))
            #     total_date_n = len(file_list)
            #     # rand_num = random.randint(int(total_date_n*self.train_rate),total_date_n-1)
            #     rand_num = int(index % (total_date_n - 1 - int(total_date_n * self.train_rate)) + (total_date_n * self.train_rate))
            #     target_filename = file_list[rand_num]
                
            price = np.load(join(PROJ_DIR, "datasets", "raw_data", "price", company, target_filename))
            tweet = np.load(join(PROJ_DIR, "datasets", "raw_data", "text", company, target_filename))
            label = np.load(join(PROJ_DIR, "datasets", "raw_data", "label", company, target_filename))
            price_list.append(price)
            tweet_list.append(tweet)
            label_list.append(label)
        
        price_list = np.array(price_list)
        tweet_list = np.array(tweet_list)
        label_list = np.array(label_list)
        
        return price_list, tweet_list, label_list


class StockDataset(Dataset):
    def __init__(self, company_list, data_num, train_rate=0.7, is_train=True):
        self.company_list = company_list
        self.data_num = data_num
        self.train_rate = train_rate
        self.is_train = is_train
        self.max_len = 0
        for company in self.company_list:
            max_len = len(os.listdir(join(PROJ_DIR, "datasets", "raw_data", "label", company)))
            self.max_len = max(max_len, self.max_len)

        
        if self.is_train:
            print('Build training data...')
        else:
            print('Build test data...')
    
    def __len__(self):
        # return self.data_num
        return self.max_len
    
    def __getitem__(self, index):
        price_list = []
        tweet_list = []
        label_list = []
        for company in self.company_list:
            if self.is_train:
                file_list = os.listdir(join(PROJ_DIR, "datasets", "raw_data", "label", company))
                total_date_n = len(file_list)
                # rand_num = random.randint(0,int(total_date_n*self.train_rate)-1)
                rand_num = int(index % (int(total_date_n * self.train_rate) - 1))
                target_filename = file_list[rand_num]
            else:
                file_list = os.listdir(join(PROJ_DIR, "datasets", "raw_data", "label", company))
                total_date_n = len(file_list)
                # rand_num = random.randint(int(total_date_n*self.train_rate),total_date_n-1)
                rand_num = int(index % (total_date_n - 1 - int(total_date_n * self.train_rate)) + (total_date_n * self.train_rate))
                target_filename = file_list[rand_num]
                
            price = np.load(join(PROJ_DIR, "datasets", "raw_data", "price", company, target_filename))
            tweet = np.load(join(PROJ_DIR, "datasets", "raw_data", "text", company, target_filename))
            label = np.load(join(PROJ_DIR, "datasets", "raw_data", "label", company, target_filename))
            price_list.append(price)
            tweet_list.append(tweet)
            label_list.append(label)
        
        price_list = np.array(price_list)
        tweet_list = np.array(tweet_list)
        label_list = np.array(label_list)
        
        return price_list, tweet_list, label_list
    

class PriceTextDataset(Dataset):
    def __init__(self, company_list, train=True, train_rate=0.7):
        self.company_list = company_list
        # self.data_dir = data_dir
        self.train = train
        self.train_rate = train_rate
        
        self.price_list = []
        self.tweet_list = []
        self.label_list = []

        if self.train:
            for company in self.company_list:
                price_files = os.listdir(join(PROJ_DIR, "datasets", "raw_data", "price", company))
                total_date_n = len(price_files)
                train_idx = int(total_date_n * self.train_rate)
                price_files = price_files[:train_idx]
                for filename in price_files:
                    price = np.load(join(PROJ_DIR, "datasets", "raw_data", "price", company, filename))
                    tweet = np.load(join(PROJ_DIR, "datasets", "raw_data", "text", company, filename))
                    label = np.load(join(PROJ_DIR, "datasets", "raw_data", "label", company, filename))
                    self.price_list.append(price)
                    self.tweet_list.append(tweet)
                    self.label_list.append(label)
        else:
            for company in self.company_list:
                price_files = os.listdir(join(PROJ_DIR, "datasets", "raw_data", "price", company))
                total_date_n = len(price_files)
                train_idx = int(total_date_n * self.train_rate)
                price_files = price_files[train_idx:]
                for filename in price_files:
                    price = np.load(join(PROJ_DIR, "datasets", "raw_data", "price", company, filename))
                    tweet = np.load(join(PROJ_DIR, "datasets", "raw_data", "text", company, filename))
                    label = np.load(join(PROJ_DIR, "datasets", "raw_data", "label", company, filename))
                    self.price_list.append(price)
                    self.tweet_list.append(tweet)
                    self.label_list.append(label)

    def __len__(self):
        return len(self.price_list)

    def __getitem__(self, idx):
        price = torch.tensor(self.price_list[idx], dtype=torch.float32)
        tweet = torch.tensor(self.tweet_list[idx], dtype=torch.float32)
        label = torch.tensor(self.label_list[idx])
        return price, tweet, label
    

class StockDataset2(Dataset):
    def __init__(self, company_list, data_num, train_rate=0.7, is_train=True):
        self.company_list = company_list
        self.data_num = data_num
        self.train_rate = train_rate
        self.is_train = is_train
    
    def __len__(self):
        return self.data_num
    
    def __getitem__(self, index):
        price_list = []
        tweet_list = []
        label_list = []
        for company in self.company_list:
            file_list = os.listdir(join(PROJ_DIR, "datasets", "raw_data", "label", company))
            total_date_n = len(file_list)
            
            if self.is_train == True:
                file_indices = range(int(total_date_n * self.train_rate))
            elif self.is_train == False:
                file_indices = range(int(total_date_n * self.train_rate), total_date_n)
            else:
                raise ValueError(f"Invalid split value:")
                
            for i in file_indices:
                target_filename = file_list[i]
                price = np.load(join(PROJ_DIR, "datasets", "raw_data", "price", company, target_filename))
                tweet = np.load(join(PROJ_DIR, "datasets", "raw_data", "text", company, target_filename))
                label = np.load(join(PROJ_DIR, "datasets", "raw_data", "label", company, target_filename))
                price_list.append(price)
                tweet_list.append(tweet)
                label_list.append(label)
        
        price_list = np.array(price_list)
        tweet_list = np.array(tweet_list)
        label_list = np.array(label_list)
        
        return price_list, tweet_list, label_list