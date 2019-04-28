import csv
from brian2 import *
import dateutil.parser
import numpy as np


def return2019Data(file):
    out = []
    with open(file) as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            date = dateutil.parser.parse(row['Date'],ignoretz=True).isoformat()
            date_split = date.split("-")
            if(date_split[0] == '2018'):
                out.append(int(float(row['Adj Close']) * 10))
    return np.asarray(out)

def return2018Data(file):
    out = []
    with open(file) as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            date = dateutil.parser.parse(row['Date'],ignoretz=True).isoformat()
            date_split = date.split("-")
            if(date_split[0] == '2017'):
                out.append(int(float(row['Adj Close']) * 10))
    return np.asarray(out)

def getMinMaxDiff(file):
    new_arr = np.concatenate((return2018Data(file), return2019Data(file)), axis=None)
    return max(new_arr) - min(new_arr)

def buildInputArray(numNeurons, data, lag=0):
    m = map(lambda d: [i == (d * second) % numNeurons for i in range(numNeurons)], data)
    indices, times = [], []
    for j, row in enumerate(m):
        for i, v in enumerate(row):
            if v:
                indices.append(i)
                times.append(j + lag)
    return indices, times
