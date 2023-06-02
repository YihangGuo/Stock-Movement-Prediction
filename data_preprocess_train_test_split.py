import tensorflow_hub as hub
import json
import numpy as np
import os
from posixpath import join
from os.path import abspath, dirname
from settings import *
from tqdm import tqdm


train_start_date = '2014-01-01'
train_end_date = '2015-07-31'
val_start_date = '2015-08-01'
val_end_date = '2015-09-30'
test_start_date = '2015-10-01'
test_end_date = '2015-12-31'


def make_dirs(raw_data_path, train_data_path, test_data_path, val_data_path):
    if not os.path.exists(raw_data_path):
        os.makedirs(raw_data_path)

    if not os.path.exists(train_data_path):
        os.makedirs(train_data_path)
    
    if not os.path.exists(test_data_path):
        os.makedirs(test_data_path)

    if not os.path.exists(val_data_path):
        os.makedirs(val_data_path)


def concate(input):
    output = ''
    for s in input:
        output += s
        if(s != '$'):
            output += ' '
    return output


def get_valid_company_list(filename):
    company_list = []
    with open(filename, 'r') as csvfile:
        csvreader = csv.reader(csvfile)
        for comp in csvreader:
            company_list.extend(comp)
    return company_list


def prep_tweets(raw_data_path, stocknet_data_path, train_data_path, test_data_path, val_data_path):
    # USE embedding method
    USE_embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")

    if not os.path.exists(join(raw_data_path, "text")):
        os.mkdir(join(raw_data_path, "text"))

    all_emb_list = []
    align_file = open(join(PROJ_DIR, "alignment_date_list.txt"))

    with open(join(PROJ_DIR, 'alignment_dates.json'), 'r') as f:
        alignment_dates = json.load(f)

    for company, dates in tqdm(alignment_dates.items()):
        test_text_dir_path = join(test_data_path, "text", company)
        if not os.path.exists(test_text_dir_path):
            os.makedirs(test_text_dir_path, exist_ok=True)
        
        train_text_dir_path = join(train_data_path, "text", company)
        if not os.path.exists(train_text_dir_path):
            os.makedirs(train_text_dir_path, exist_ok=True)

        val_text_dir_path = join(val_data_path, "text", company)
        if not os.path.exists(val_text_dir_path):
            os.makedirs(val_text_dir_path, exist_ok=True)

        emb_dict = {}

        for date_window in dates:
            target_date = date_window[0]
            day_emb_list = np.zeros((5, TWEET_NUM, 512)) # TWEET_NUM = 8
            if target_date >= train_start_date and target_date <= train_end_date:
                for i in range(1, 6):
                    emb_list = np.zeros((TWEET_NUM, 512))
                    if not date_window[i] in emb_dict.keys():
                        file = open(join(stocknet_data_path, "tweet", "preprocessed", company, date_window[i]),'r')
                        idx = 0
                        for l in file.readlines():
                            output = json.loads(l)
                            output = concate(output['text'])
                            emb_output = USE_embed([output]).numpy()[0]
                            emb_list[idx] = emb_output
                            idx += 1
                            if(idx >= TWEET_NUM):
                                break
                        emb_dict[date_window[i]] = emb_list
                    else:
                        emb_list = emb_dict[date_window[i]]
                    day_emb_list[i-1] = emb_list
                np.save(join(train_text_dir_path, date_window[0]+'.npy'), day_emb_list)
            elif target_date >= test_start_date and target_date <= test_end_date:
                for i in range(1, 6):
                    emb_list = np.zeros((TWEET_NUM, 512))
                    if not date_window[i] in emb_dict.keys():
                        file = open(join(stocknet_data_path, "tweet", "preprocessed", company, date_window[i]),'r')
                        idx = 0
                        for l in file.readlines():
                            output = json.loads(l)
                            output = concate(output['text'])
                            emb_output = USE_embed([output]).numpy()[0]
                            emb_list[idx] = emb_output
                            idx += 1
                            if(idx >= TWEET_NUM):
                                break
                        emb_dict[date_window[i]] = emb_list
                    else:
                        emb_list = emb_dict[date_window[i]]
                    day_emb_list[i-1] = emb_list
                np.save(join(test_text_dir_path, date_window[0]+'.npy'), day_emb_list)
            elif target_date >= val_start_date and target_date <= val_end_date:
                for i in range(1, 6):
                    emb_list = np.zeros((TWEET_NUM, 512))
                    if not date_window[i] in emb_dict.keys():
                        file = open(join(stocknet_data_path, "tweet", "preprocessed", company, date_window[i]),'r')
                        idx = 0
                        for l in file.readlines():
                            output = json.loads(l)
                            output = concate(output['text'])
                            emb_output = USE_embed([output]).numpy()[0]
                            emb_list[idx] = emb_output
                            idx += 1
                            if(idx >= TWEET_NUM):
                                break
                        emb_dict[date_window[i]] = emb_list
                    else:
                        emb_list = emb_dict[date_window[i]]
                    day_emb_list[i-1] = emb_list
                np.save(join(val_text_dir_path, date_window[0]+'.npy'), day_emb_list)
        print("=== Company:",company," OK ===")
    align_file.close()


def get_price_data(dir_path, company_list, split=False):
    # read raw data
    path = os.path.join(dir_path,"stocknet-dataset","price","raw")

    dataset = []
    for company in company_list:
        last_adj_close = 1
        data = []
        file_path = os.path.join(path, company + ".csv")
        file = open(file_path)
        line = file.readline()
        while True:
            line = file.readline()
            line = line.split('\n')
            if(line[0] == ''):
                break
            line = line[0].split(',')
            if(line[0] >= '2013-12-24' and line[0] <= '2016-01-01'):
                feature = []
                feature.append(line[0])
                feature.append(float(line[2])/last_adj_close) # high
                feature.append(float(line[3])/last_adj_close) # low
                feature.append(float(line[5])/last_adj_close) # adj close
                data.append(feature)
            elif(line[0] >= '2016-01-01'):
                break
            last_adj_close = float(line[5])
        file.close()
        dataset.append(data)
        
    # Add label
    idx = 0
    for data in dataset:
        for i in range(5,len(data)):
            if(data[i][3]-1 >= 0): # adj close / last adj close - 1 >= 0.0055
                data[i].append(1)
            elif(data[i][3]-1 < 0): # adj close / last adj close - 1 <= -0.005
                data[i].append(0)

    # Pickup useful data
    database = []
    for data in dataset:
        table = []
        for i in range(5,len(data)):
            if(len(data[i]) >= 5):
                dict_ = {}
                dict_["date"] = data[i][0]
                feat = []
                feat.append(data[i-1][1:4])
                feat.append(data[i-2][1:4])
                feat.append(data[i-3][1:4])
                feat.append(data[i-4][1:4])
                feat.append(data[i-5][1:4])
                dict_["feature"] = np.array(feat)
                dict_["label"] = data[i][4]
                table.append(dict_) 
        database.append(table)
    return database


def build_date_dict(n):
    data_dict = {}
    dataset = database[n]
    for i in range(len(dataset)):
        date = dataset[i]['date']
        feature = dataset[i]['feature']
        data_dict[date] = feature
    return data_dict


def build_label_dict(n):
    data_dict = {}
    dataset = database[n]
    for i in range(len(dataset)):
        date = dataset[i]['date']
        label = dataset[i]['label']
        data_dict[date] = label
    return data_dict


if __name__ == '__main__':
    # create diretory
    raw_data_path = join(PROJ_DIR, "datasets", "raw_data")
    train_data_path = join(PROJ_DIR, "datasets", "train")
    test_data_path = join(PROJ_DIR, "datasets", "test")
    val_data_path = join(PROJ_DIR, "datasets", "val")
    make_dirs(raw_data_path, train_data_path, test_data_path, val_data_path)

    company_list = get_valid_company_list(join(PROJ_DIR, "valid_company.csv"))

    # prep tweets
    stocknet_data_path = join(PROJ_DIR, "datasets", "stocknet-dataset")
    prep_tweets(raw_data_path, stocknet_data_path, train_data_path, test_data_path, val_data_path)

    # Price & Label Preprocessing
    database = get_price_data(join(PROJ_DIR, "datasets"), company_list)
    assert(len(database) == 30)

    with open(join(PROJ_DIR, 'alignment_dates.json'), 'r') as f:
        alignment_dates = json.load(f)

    for n, (company, dates) in tqdm(enumerate(alignment_dates.items())):
        test_price_dir_path = join(test_data_path, "price", company)
        if not os.path.exists(test_price_dir_path):
            os.makedirs(test_price_dir_path, exist_ok=True)

        test_label_dir_path = join(test_data_path, "label", company)
        if not os.path.exists(test_label_dir_path):
            os.makedirs(test_label_dir_path, exist_ok=True)

        train_price_dir_path = join(train_data_path, "price", company)
        if not os.path.exists(train_price_dir_path):
            os.makedirs(train_price_dir_path, exist_ok=True)

        train_label_dir_path = join(train_data_path, "label", company)
        if not os.path.exists(train_label_dir_path):
            os.makedirs(train_label_dir_path, exist_ok=True)

        val_price_dir_path = join(val_data_path, "price", company)
        if not os.path.exists(val_price_dir_path):
            os.makedirs(val_price_dir_path, exist_ok=True)

        val_label_dir_path = join(val_data_path, "label", company)
        if not os.path.exists(val_label_dir_path):
            os.makedirs(val_label_dir_path, exist_ok=True)
        
        
        date_dict = build_date_dict(n)
        label_dict = build_label_dict(n)
        for date_window in dates:
            target_date = date_window[0]
            if target_date >= train_start_date and target_date <= train_end_date:
                feature = date_dict[target_date]
                np.save(join(train_price_dir_path, target_date + '.npy'), feature)

                label = label_dict[target_date]
                np.save(join(train_label_dir_path, target_date + '.npy'), label)
            elif target_date >= test_start_date and target_date <= test_end_date:
                feature = date_dict[target_date]
                np.save(join(test_price_dir_path, target_date + '.npy'), feature)

                label = label_dict[target_date]
                np.save(join(test_label_dir_path, target_date + '.npy'), label)
            elif target_date >= val_start_date and target_date <= val_end_date:
                feature = date_dict[target_date]
                np.save(join(val_price_dir_path, target_date + '.npy'), feature)

                label = label_dict[target_date]
                np.save(join(val_label_dir_path, target_date + '.npy'), label)

        print("=== Company:",company," OK ===")
    print("====== Creating raw data is done ======")

# date_dict['2014-01-02'] shape= (5.3) 
# array([[1.08785502, 1.07374523, 1.01172181],
#    [1.07475299, 1.05984318, 0.99005514],
#    [1.07572506, 1.06636702, 0.99324354],
#    [1.0782179 , 1.06663112, 0.99335899],
#    [1.07812761, 1.06709895, 0.99575484]])

# label_dict['2014-01-02'] 0/1

