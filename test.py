# test file 
import numpy as np
import csv
import pandas as pd

# test fixed row size in csv
def insert():
    m = 10
    # create numpy array with 1000 rows and 70 columns with type float32
    embeds = np.random.random((m, 70)).astype(np.float32)
    # add id column to the numpy array starting from 1
    ids = np.arange(1, m+1).reshape(-1, 1)
    # concatenate the id column to the numpy array
    embeds = np.concatenate((ids, embeds), axis=1).astype(np.float32)
    # open csv file and write the numpy array to it as numbers after last row without empty line
    with open("test.csv", "a+") as fout:
        np.savetxt(fout, embeds, delimiter=",", fmt="%f")
def load():
    # load csv file as numpy array
    embeds = np.loadtxt("test.csv", delimiter=",")
    # get the id column
    ids = embeds[:, 0]
    # get the embed columns
    embeds = embeds[:, 1:]
    # print the numpy array
    print("embeddings",embeds)
    print("ids",ids)
    print("shape of embeddings",embeds.shape)
    print("shape of ids",ids.shape)
    print("type of embeddings",embeds.dtype)
    print("type of ids",ids.dtype)
    



def main():
    # print("Hello World!")
    insert()
    load()
if __name__ == "__main__":
    main()
    