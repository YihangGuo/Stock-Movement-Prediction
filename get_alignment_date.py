import json
from settings import *


def apply_6_day_window(matching_dates):
    result = {}
    for stock, dates in matching_dates.items():
        if len(dates) >= 6:
            result[stock] = []
            for i in range(len(dates) - 5):
                window = dates[i:i+6]
                window.reverse()
                result[stock].append(window)
    return result

def save_windowed_dates(input_file, output_file):
    with open(input_file, 'r') as file:
        matching_dates = json.load(file)
    
    windowed_dates = apply_6_day_window(matching_dates)
    
    with open(output_file, 'w') as file:
        json.dump(windowed_dates, file)

input_file = join(PROJ_DIR, 'matching_dates.json')
output_file = join(PROJ_DIR, 'alignment_dates.json')

save_windowed_dates(input_file, output_file)