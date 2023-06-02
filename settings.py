import os
from os.path import abspath, dirname, join, exists
from collections import defaultdict
import json
import codecs
import csv
from tqdm import tqdm
import pickle
import random
import numpy as np
import torch

PROJ_DIR = abspath(dirname(__file__))

TRAIN_SIZE = 300
TRAIN_RATE = 0.7
TEST_SIZE = 90

TWEET_NUM = 8


# train_start_date = '2014-01-01'
# train_end_date = '2015-07-31'
# val_start_date = '2015-08-01'
# val_end_date = '2015-09-30'
# test_start_date = '2015-10-01'
# test_end_date = '2016-01-01'
