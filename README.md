# Stock Movement Prediction Project

## Introduction
[Deep Attentive Learning for Stock Movement Prediction
From Social Media Text and Company Correlations](https://aclanthology.org/2020.emnlp-main.676.pdf)

## Installation
### Requirements
- `conda create -n stock-pred python=3.7`
- `conda install pytorch`
- `pip install -r requirements.txt`
- `conda activate stock-pred`

### Steps
1. Prepare the datasets, the [stocknet-dataset](https://github.com/yumoxu/stocknet-dataset) should be downloaded under datasets folder, [relation](https://github.com/fulifeng/Temporal_Relational_Stock_Ranking/tree/master/data) folder should be download under datasets folder.

2. To run the code, first run the data_preprocess scipt,
- `python data_preprocess_train_test_split.py`

3. Then, start the run & evaluation process by,
- `python run.py`

4. The checkpoint/log/plots information will be stored under checkpoints/log/plots
