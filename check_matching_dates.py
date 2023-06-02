import os
import csv
from settings import *
from posixpath import join
import json


def check_matching_files(stock_file, tweets_folder, stock_name):
    # Read stock data from the CSV file
    stock_data = []
    with open(stock_file, 'r') as file:
        csv_reader = csv.reader(file)
        next(csv_reader)  # Skip header row
        for row in csv_reader:
            stock_data.append(row[0])  # Store only the date
    
    # Find matching files in the tweets folder
    matching_files = []
    for file_name in os.listdir(join(tweets_folder, stock_name)):
            file_date = file_name  # Remove extension to get the date
            if file_date in stock_data:
                matching_files.append(file_date)
    
    matching_files.sort()
    
    return matching_files


def save_matching_dates(stock_folder, tweets_folder, output_file):
    company_list = get_valid_company_list(join(PROJ_DIR, "valid_company.csv"))
    matching_dates = {}

    for stock_name in company_list:

        stock_file = stock_name + '.csv'
        stock_path = os.path.join(stock_folder, stock_file)
        
        # Check matching files for each stock
        matching_files = check_matching_files(stock_path, tweets_folder, stock_name)
        matching_dates[stock_name] = matching_files


    with open(output_file, 'w') as file:
        json.dump(matching_dates, file)


def get_valid_company_list(filename):
    company_list = []

    with open(filename, 'r') as csvfile:
        # Create a CSV reader object
        csvreader = csv.reader(csvfile)
        # Loop over each row in the CSV file
        for comp in csvreader:
            # Append the row to the list of rows
            company_list.extend(comp)
    return company_list


stock_folder = join(PROJ_DIR, "datasets", "stocknet-dataset", "price", "raw")
tweets_folder = join(PROJ_DIR, "datasets", "stocknet-dataset", "tweet", "raw")
output_file = join(PROJ_DIR, 'matching_dates.json')

save_matching_dates(stock_folder, tweets_folder, output_file)