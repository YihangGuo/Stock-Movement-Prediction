import csv
from posixpath import join
from os.path import abspath, dirname
PROJ_DIR = abspath(dirname(__file__))

""" # Open the input text file and the output CSV file
with open(join(PROJ_DIR, "valid_company.txt"), 'r') as infile, open(join(PROJ_DIR, 'valid_company.csv'), 'w', newline='') as outfile:
    # Create a CSV writer object
    writer = csv.writer(outfile)
    # Loop over each line in the input file
    for line in infile:
        # Remove any trailing whitespace from the line
        line = line.strip()
        # Split the line into a list of values
        values = line.split()
        # Write the values to a new row in the CSV file
        writer.writerow(values) """

def get_valid_company_list(filename):
    company_list = []
    # Open the CSV file
    with open(filename, 'r') as csvfile:
        # Create a CSV reader object
        csvreader = csv.reader(csvfile)
        # Loop over each row in the CSV file
        for comp in csvreader:
            # Append the row to the list of rows
            company_list.extend(comp)
    return company_list

def get_valid_company_list_txt(filename):
    company_list = []

    file = open(filename)
    while True:
        line = file.readline()
        line = line.replace("\n","")
        if(line == ""):
            break
        company_list.append(line)
    file.close()
    return company_list


if __name__ == '__main__':
    cs = get_valid_company_list(join(PROJ_DIR, "valid_company.csv"))
    print(cs)

    cs2 = get_valid_company_list_txt(join(PROJ_DIR, "valid_company.csv"))
    print(cs2)