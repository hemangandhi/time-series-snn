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
            if(date_split[0] == '1963'):
                out.append(float(row['Adj Close']))
    return np.asarray(out)

def return2018Data(file):
    out = []
    with open(file) as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            date = dateutil.parser.parse(row['Date'],ignoretz=True).isoformat()
            date_split = date.split("-")
            if(date_split[0] == '1962'):
                out.append(float(row['Adj Close']))
    return np.asarray(out)

def getMinMaxDiff(file):
    new_arr = np.concatenate((return2018Data(file), return2019Data(file)), axis=None)
    return max(new_arr) - min(new_arr)

def buildInputArray(numNeurons, data, lag=0):
    data_min = min(data)
    data_max = max(data)
    def bucket_of_datum(datum):
        bucket_size = (numNeurons - 1) / (data_max - data_min)
        return int(bucket_size * (datum - data_min))

    m = map(bucket_of_datum, data)
    indices, times = [], []
    for j, i in enumerate(m):
        indices.append(i)
        times.append(j + lag)
    return indices, times

if __name__ == "__main__":
    FILE = "data/IBM.csv"
    daddy_bezos = return2018Data(FILE) * Hz
    print(buildInputArray(100, daddy_bezos))
