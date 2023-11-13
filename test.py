# test file 
import numpy as np
import csv
import pandas as pd
import os
import subprocess
import time

def get_number_of_rows(file_path):
    result = subprocess.run(['find', '/c', '/v', '""', '<', file_path], stdout=subprocess.PIPE, text=True)
    # Extract the number of lines from the command output
    num_rows = int(result.stdout.strip().split('\n')[-1].split(' ')[-1])
    return num_rows
def get_number_of_rows_linux(file_path):
    result = subprocess.run(['wc', '-l', file_path], stdout=subprocess.PIPE, text=True)
    return int(result.stdout.split()[0])

# test fixed row size in csv
def insert():

    m = 1000000
    # create numpy array with 1000 rows and 70 columns with type float32
    embeds = np.random.random((m, 70)).astype(np.float32)
    # add id column to the numpy array starting from 1
    ids = np.arange(1, m+1).reshape(-1, 1)
    # concatenate the id column to the numpy array
    embeds = np.concatenate((ids, embeds), axis=1).astype(np.float32)
    # open csv file and write the numpy array to it as numbers after last row without empty line
    with open("test.csv", "w") as fout:
        np.savetxt(fout, embeds, delimiter=",", fmt="%f")
def load():

    start_time = time.time()
    print("Number of rows in csv file: ", get_number_of_rows("./test.csv"))
    print("time to get number of rows = ", time.time() - start_time)
    with open("test.csv", "a+") as fout:
        # get end of file byte offset
        # fout.seek(0, 2)
        # get the byte offset of the last row
        last_row_byte_offset = fout.tell()
        print("last_row_byte_offset = ",last_row_byte_offset)
        # read the last row
        fout.seek(last_row_byte_offset - 80*8)
        # read the last row
        specific_row = fout.readline().strip()
        print(specific_row)

    # # load csv file as numpy array
    # embeds = np.loadtxt("test.csv", delimiter=",")
    # # get the id column
    # ids = embeds[:, 0]
    # # get the embed columns
    # embeds = embeds[:, 1:]
    # # print the numpy array
    # # print("embeddings",embeds)
    # print("ids",ids)
    # # print("shape of embeddings",embeds.shape)
    # # print("shape of ids",ids.shape)
    # # print("type of embeddings",embeds.dtype)
    # # print("type of ids",ids.dtype)
    # file_size = os.path.getsize("test.csv")
    # num_lines = file_size // (80*8)
    # print("Number of lines in csv file: ", num_lines)
    
    



def main():
    # print("Hello World!")
    insert()
    load()
if __name__ == "__main__":
    main()
    