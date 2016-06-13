import csv
import json
import preprocessing.preprocessing_parameters as pp

#This file is used to read the original stackexchange dataset (format = csv)
#and convert it into .json format 
def readRawCSVFileAndConvertItIntoJsonFile(input_file, output_file):
    
    """
    Reads a raw .csv file and converts it into json file.
    
    :param input_file - Name of input file (located in #pp.RESOURCES_FOLDER)
    :param output_file - Name of output file (written to #pp.RESOURCES_FOLDER)
    """
    
    csv_file = open(pp.RESOURCES_FOLDER + input_file, 'r')
    json_file = open(pp.RESOURCES_FOLDER + output_file, 'w')
    
    reader = csv.DictReader(csv_file, pp.STACKEXCHANGE_DATA_COLUMNS)
    
    #convert reader to list (to count num_rows)
    rows = list(reader)
    
    row_count = 0
    
    for row in rows:
        row_count += 1
        json.dump(row, json_file)
        json_file.write('\n')